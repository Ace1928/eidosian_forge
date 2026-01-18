from numba.core import types, errors, ir, sigutils, ir_utils
from numba.core.typing.typeof import typeof_impl
from numba.core.transforms import find_region_inout_vars
from numba.core.ir_utils import build_definitions
import numba
def mutate_with_body(self, func_ir, blocks, blk_start, blk_end, body_blocks, dispatcher_factory, extra):
    ir_utils.dprint_func_ir(func_ir, 'Before with changes', blocks=blocks)
    assert extra is not None
    args = extra['args']
    assert len(args) == 1
    arg = args[0]
    scope = blocks[blk_start].scope
    loc = blocks[blk_start].loc
    if isinstance(arg, ir.Arg):
        arg = ir.Var(scope, arg.name, loc)
    set_state = []
    restore_state = []
    gvar = scope.redefine('$ngvar', loc)
    set_state.append(ir.Assign(ir.Global('numba', numba, loc), gvar, loc))
    spcattr = ir.Expr.getattr(gvar, 'set_parallel_chunksize', loc)
    spcvar = scope.redefine('$spc', loc)
    set_state.append(ir.Assign(spcattr, spcvar, loc))
    orig_pc_var = scope.redefine('$save_pc', loc)
    cs_var = scope.redefine('$cs_var', loc)
    set_state.append(ir.Assign(arg, cs_var, loc))
    spc_call = ir.Expr.call(spcvar, [cs_var], (), loc)
    set_state.append(ir.Assign(spc_call, orig_pc_var, loc))
    restore_spc_call = ir.Expr.call(spcvar, [orig_pc_var], (), loc)
    restore_state.append(ir.Assign(restore_spc_call, orig_pc_var, loc))
    blocks[blk_start].body = blocks[blk_start].body[1:-1] + set_state + [blocks[blk_start].body[-1]]
    blocks[blk_end].body = restore_state + blocks[blk_end].body
    func_ir._definitions = build_definitions(blocks)
    ir_utils.dprint_func_ir(func_ir, 'After with changes', blocks=blocks)