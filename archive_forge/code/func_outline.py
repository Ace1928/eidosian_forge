from pythran.analyses import (ImportedIds, HasReturn, IsAssigned, CFG,
from pythran.passmanager import Transformation
from pythran.syntax import PythranSyntaxError
import gast as ast
from copy import deepcopy
def outline(name, formal_parameters, out_parameters, stmts, has_return, has_break, has_cont):
    args = ast.arguments([ast.Name(fp, ast.Param(), None, None) for fp in formal_parameters], [], None, [], [], None, [])
    if isinstance(stmts, ast.expr):
        assert not out_parameters, 'no out parameters with expr'
        fdef = ast.FunctionDef(name, args, [ast.Return(stmts)], [], None, None)
    else:
        fdef = ast.FunctionDef(name, args, stmts, [], None, None)
        stmts.append(ast.Return(ast.Tuple([ast.Name(fp, ast.Load(), None, None) for fp in out_parameters], ast.Load())))
        if has_return:
            pr = PatchReturn(stmts[-1], has_break or has_cont)
            pr.visit(fdef)
        if has_break or has_cont:
            if not has_return:
                stmts[-1].value = ast.Tuple([ast.Constant(LOOP_NONE, None), stmts[-1].value], ast.Load())
            pbc = PatchBreakContinue(stmts[-1])
            pbc.visit(fdef)
    return fdef