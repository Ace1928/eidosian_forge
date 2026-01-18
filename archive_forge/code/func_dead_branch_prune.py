import operator
from functools import reduce
from collections import namedtuple, defaultdict
from .controlflow import CFGraph
from numba.core import types, errors, ir, consts
from numba.misc import special
def dead_branch_prune(func_ir, called_args):
    """
    Removes dead branches based on constant inference from function args.
    This directly mutates the IR.

    func_ir is the IR
    called_args are the actual arguments with which the function is called
    """
    from numba.core.ir_utils import get_definition, guard, find_const, GuardException
    DEBUG = 0

    def find_branches(func_ir):
        branches = []
        for blk in func_ir.blocks.values():
            branch_or_jump = blk.body[-1]
            if isinstance(branch_or_jump, ir.Branch):
                branch = branch_or_jump
                pred = guard(get_definition, func_ir, branch.cond.name)
                if pred is not None and getattr(pred, 'op', None) == 'call':
                    function = guard(get_definition, func_ir, pred.func)
                    if function is not None and isinstance(function, ir.Global) and (function.value is bool):
                        condition = guard(get_definition, func_ir, pred.args[0])
                        if condition is not None:
                            branches.append((branch, condition, blk))
        return branches

    def do_prune(take_truebr, blk):
        keep = branch.truebr if take_truebr else branch.falsebr
        jmp = ir.Jump(keep, loc=branch.loc)
        blk.body[-1] = jmp
        return 1 if keep == branch.truebr else 0

    def prune_by_type(branch, condition, blk, *conds):
        lhs_cond, rhs_cond = conds
        lhs_none = isinstance(lhs_cond, types.NoneType)
        rhs_none = isinstance(rhs_cond, types.NoneType)
        if lhs_none or rhs_none:
            try:
                take_truebr = condition.fn(lhs_cond, rhs_cond)
            except Exception:
                return (False, None)
            if DEBUG > 0:
                kill = branch.falsebr if take_truebr else branch.truebr
                print('Pruning %s' % kill, branch, lhs_cond, rhs_cond, condition.fn)
            taken = do_prune(take_truebr, blk)
            return (True, taken)
        return (False, None)

    def prune_by_value(branch, condition, blk, *conds):
        lhs_cond, rhs_cond = conds
        try:
            take_truebr = condition.fn(lhs_cond, rhs_cond)
        except Exception:
            return (False, None)
        if DEBUG > 0:
            kill = branch.falsebr if take_truebr else branch.truebr
            print('Pruning %s' % kill, branch, lhs_cond, rhs_cond, condition.fn)
        taken = do_prune(take_truebr, blk)
        return (True, taken)

    def prune_by_predicate(branch, pred, blk):
        try:
            if not isinstance(pred, (ir.Const, ir.FreeVar, ir.Global)):
                raise TypeError('Expected constant Numba IR node')
            take_truebr = bool(pred.value)
        except TypeError:
            return (False, None)
        if DEBUG > 0:
            kill = branch.falsebr if take_truebr else branch.truebr
            print('Pruning %s' % kill, branch, pred)
        taken = do_prune(take_truebr, blk)
        return (True, taken)

    class Unknown(object):
        pass

    def resolve_input_arg_const(input_arg_idx):
        """
        Resolves an input arg to a constant (if possible)
        """
        input_arg_ty = called_args[input_arg_idx]
        if isinstance(input_arg_ty, types.NoneType):
            return input_arg_ty
        if isinstance(input_arg_ty, types.Omitted):
            val = input_arg_ty.value
            if isinstance(val, types.NoneType):
                return val
            elif val is None:
                return types.NoneType('none')
        return getattr(input_arg_ty, 'literal_type', Unknown())
    if DEBUG > 1:
        print('before'.center(80, '-'))
        print(func_ir.dump())
    phi2lbl = dict()
    phi2asgn = dict()
    for lbl, blk in func_ir.blocks.items():
        for stmt in blk.body:
            if isinstance(stmt, ir.Assign):
                if isinstance(stmt.value, ir.Expr) and stmt.value.op == 'phi':
                    phi2lbl[stmt.value] = lbl
                    phi2asgn[stmt.value] = stmt
    branch_info = find_branches(func_ir)
    nullified_conditions = []
    for branch, condition, blk in branch_info:
        const_conds = []
        if isinstance(condition, ir.Expr) and condition.op == 'binop':
            prune = prune_by_value
            for arg in [condition.lhs, condition.rhs]:
                resolved_const = Unknown()
                arg_def = guard(get_definition, func_ir, arg)
                if isinstance(arg_def, ir.Arg):
                    resolved_const = resolve_input_arg_const(arg_def.index)
                    prune = prune_by_type
                else:
                    try:
                        resolved_const = find_const(func_ir, arg)
                        if resolved_const is None:
                            resolved_const = types.NoneType('none')
                    except GuardException:
                        pass
                if not isinstance(resolved_const, Unknown):
                    const_conds.append(resolved_const)
            if len(const_conds) == 2:
                prune_stat, taken = prune(branch, condition, blk, *const_conds)
                if prune_stat:
                    nullified_conditions.append(nullified(condition, taken, True))
        else:
            resolved_const = Unknown()
            try:
                pred_call = get_definition(func_ir, branch.cond)
                resolved_const = find_const(func_ir, pred_call.args[0])
                if resolved_const is None:
                    resolved_const = types.NoneType('none')
            except GuardException:
                pass
            if not isinstance(resolved_const, Unknown):
                prune_stat, taken = prune_by_predicate(branch, condition, blk)
                if prune_stat:
                    nullified_conditions.append(nullified(condition, taken, False))
    deadcond = [x.condition for x in nullified_conditions]
    for _, cond, blk in branch_info:
        if cond in deadcond:
            for x in blk.body:
                if isinstance(x, ir.Assign) and x.value is cond:
                    nullified_info = nullified_conditions[deadcond.index(cond)]
                    if nullified_info.rewrite_stmt:
                        branch_bit = nullified_info.taken_br
                        x.value = ir.Const(branch_bit, loc=x.loc)
                        defns = func_ir._definitions[x.target.name]
                        repl_idx = defns.index(cond)
                        defns[repl_idx] = x.value
    new_cfg = compute_cfg_from_blocks(func_ir.blocks)
    dead_blocks = new_cfg.dead_nodes()
    for phi, lbl in phi2lbl.items():
        if lbl in dead_blocks:
            continue
        new_incoming = [x[0] for x in new_cfg.predecessors(lbl)]
        if set(new_incoming) != set(phi.incoming_blocks):
            if len(new_incoming) == 1:
                idx = phi.incoming_blocks.index(new_incoming[0])
                phi2asgn[phi].value = phi.incoming_values[idx]
            else:
                ic_val_tmp = []
                ic_blk_tmp = []
                for ic_val, ic_blk in zip(phi.incoming_values, phi.incoming_blocks):
                    if ic_blk in dead_blocks:
                        continue
                    else:
                        ic_val_tmp.append(ic_val)
                        ic_blk_tmp.append(ic_blk)
                phi.incoming_values.clear()
                phi.incoming_values.extend(ic_val_tmp)
                phi.incoming_blocks.clear()
                phi.incoming_blocks.extend(ic_blk_tmp)
    for dead in dead_blocks:
        del func_ir.blocks[dead]
    if nullified_conditions:
        func_ir._consts = consts.ConstantInference(func_ir)
    if DEBUG > 1:
        print('after'.center(80, '-'))
        print(func_ir.dump())