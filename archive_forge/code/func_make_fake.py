from pythran.analyses import (ImportedIds, HasReturn, IsAssigned, CFG,
from pythran.passmanager import Transformation
from pythran.syntax import PythranSyntaxError
import gast as ast
from copy import deepcopy
@staticmethod
def make_fake(stmts):
    return ast.If(ast.Constant(0, None), stmts, [])