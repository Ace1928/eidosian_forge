import types as pytypes  # avoid confusion with numba.types
import copy
import ctypes
import numba.core.analysis
from numba.core import types, typing, errors, ir, rewrites, config, ir_utils
from numba.parfors.parfor import internal_prange
from numba.core.ir_utils import (
from numba.core.analysis import (
from numba.core import postproc
from numba.np.unsafe.ndarray import empty_inferred as unsafe_empty_inferred
import numpy as np
import operator
import numba.misc.special
def inline_closure_call(func_ir, glbls, block, i, callee, typingctx=None, targetctx=None, arg_typs=None, typemap=None, calltypes=None, work_list=None, callee_validator=None, replace_freevars=True):
    """Inline the body of `callee` at its callsite (`i`-th instruction of
    `block`)

    `func_ir` is the func_ir object of the caller function and `glbls` is its
    global variable environment (func_ir.func_id.func.__globals__).
    `block` is the IR block of the callsite and `i` is the index of the
    callsite's node. `callee` is either the called function or a
    make_function node. `typingctx`, `typemap` and `calltypes` are typing
    data structures of the caller, available if we are in a typed pass.
    `arg_typs` includes the types of the arguments at the callsite.
    `callee_validator` is an optional callable which can be used to validate the
    IR of the callee to ensure that it contains IR supported for inlining, it
    takes one argument, the func_ir of the callee

    Returns IR blocks of the callee and the variable renaming dictionary used
    for them to facilitate further processing of new blocks.
    """
    scope = block.scope
    instr = block.body[i]
    call_expr = instr.value
    debug_print = _make_debug_print('inline_closure_call')
    debug_print('Found closure call: ', instr, ' with callee = ', callee)
    callee_code = callee.code if hasattr(callee, 'code') else callee.__code__
    callee_closure = callee.closure if hasattr(callee, 'closure') else callee.__closure__
    if isinstance(callee, pytypes.FunctionType):
        from numba.core import compiler
        callee_ir = compiler.run_frontend(callee, inline_closures=True)
    else:
        callee_ir = get_ir_of_code(glbls, callee_code)
    if callee_validator is not None:
        callee_validator(callee_ir)
    callee_blocks = callee_ir.blocks
    max_label = max(ir_utils._the_max_label.next(), max(func_ir.blocks.keys()))
    callee_blocks = add_offset_to_labels(callee_blocks, max_label + 1)
    callee_blocks = simplify_CFG(callee_blocks)
    callee_ir.blocks = callee_blocks
    min_label = min(callee_blocks.keys())
    max_label = max(callee_blocks.keys())
    ir_utils._the_max_label.update(max_label)
    debug_print('After relabel')
    _debug_dump(callee_ir)
    callee_scopes = _get_all_scopes(callee_blocks)
    debug_print('callee_scopes = ', callee_scopes)
    assert len(callee_scopes) == 1
    callee_scope = callee_scopes[0]
    var_dict = {}
    for var in callee_scope.localvars._con.values():
        if not var.name in callee_code.co_freevars:
            inlined_name = _created_inlined_var_name(callee_ir.func_id.unique_name, var.name)
            new_var = scope.redefine(inlined_name, loc=var.loc)
            var_dict[var.name] = new_var
    debug_print('var_dict = ', var_dict)
    replace_vars(callee_blocks, var_dict)
    debug_print('After local var rename')
    _debug_dump(callee_ir)
    args = _get_callee_args(call_expr, callee, block.body[i].loc, func_ir)
    debug_print('After arguments rename: ')
    _debug_dump(callee_ir)
    if callee_closure and replace_freevars:
        closure = func_ir.get_definition(callee_closure)
        debug_print("callee's closure = ", closure)
        if isinstance(closure, tuple):
            cellget = ctypes.pythonapi.PyCell_Get
            cellget.restype = ctypes.py_object
            cellget.argtypes = (ctypes.py_object,)
            items = tuple((cellget(x) for x in closure))
        else:
            assert isinstance(closure, ir.Expr) and closure.op == 'build_tuple'
            items = closure.items
        assert len(callee_code.co_freevars) == len(items)
        _replace_freevars(callee_blocks, items)
        debug_print('After closure rename')
        _debug_dump(callee_ir)
    if typingctx:
        from numba.core import typed_passes
        callee_ir._definitions = ir_utils.build_definitions(callee_ir.blocks)
        numba.core.analysis.dead_branch_prune(callee_ir, arg_typs)
        try:
            [f_typemap, f_return_type, f_calltypes, _] = typed_passes.type_inference_stage(typingctx, targetctx, callee_ir, arg_typs, None)
        except Exception:
            [f_typemap, f_return_type, f_calltypes, _] = typed_passes.type_inference_stage(typingctx, targetctx, callee_ir, arg_typs, None)
        canonicalize_array_math(callee_ir, f_typemap, f_calltypes, typingctx)
        arg_names = [vname for vname in f_typemap if vname.startswith('arg.')]
        for a in arg_names:
            f_typemap.pop(a)
        typemap.update(f_typemap)
        calltypes.update(f_calltypes)
    _replace_args_with(callee_blocks, args)
    new_blocks = []
    new_block = ir.Block(scope, block.loc)
    new_block.body = block.body[i + 1:]
    new_label = next_label()
    func_ir.blocks[new_label] = new_block
    new_blocks.append((new_label, new_block))
    block.body = block.body[:i]
    block.body.append(ir.Jump(min_label, instr.loc))
    topo_order = find_topo_order(callee_blocks)
    _replace_returns(callee_blocks, instr.target, new_label)
    if instr.target.name in func_ir._definitions and call_expr in func_ir._definitions[instr.target.name]:
        func_ir._definitions[instr.target.name].remove(call_expr)
    for label in topo_order:
        block = callee_blocks[label]
        block.scope = scope
        _add_definitions(func_ir, block)
        func_ir.blocks[label] = block
        new_blocks.append((label, block))
    debug_print('After merge in')
    _debug_dump(func_ir)
    if work_list is not None:
        for block in new_blocks:
            work_list.append(block)
    return (callee_blocks, var_dict)