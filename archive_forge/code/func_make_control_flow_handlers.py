from pythran.analyses import (ImportedIds, HasReturn, IsAssigned, CFG,
from pythran.passmanager import Transformation
from pythran.syntax import PythranSyntaxError
import gast as ast
from copy import deepcopy
def make_control_flow_handlers(self, cont_n, status_n, expected_return, has_cont, has_break):
    """
        Create the statements in charge of gathering control flow information
        for the static_if result, and executes the expected control flow
        instruction
        """
    if expected_return:
        assign = cont_ass = [ast.Assign([ast.Tuple(expected_return, ast.Store())], ast.Name(cont_n, ast.Load(), None, None), None)]
    else:
        assign = cont_ass = []
    if has_cont:
        cmpr = ast.Compare(ast.Name(status_n, ast.Load(), None, None), [ast.Eq()], [ast.Constant(LOOP_CONT, None)])
        cont_ass = [ast.If(cmpr, deepcopy(assign) + [ast.Continue()], cont_ass)]
    if has_break:
        cmpr = ast.Compare(ast.Name(status_n, ast.Load(), None, None), [ast.Eq()], [ast.Constant(LOOP_BREAK, None)])
        cont_ass = [ast.If(cmpr, deepcopy(assign) + [ast.Break()], cont_ass)]
    return cont_ass