from pythran.passmanager import FunctionAnalysis
from pythran.tables import MODULES
import gast as ast
def is_safe_use(self, use):
    parent = self.ancestors[use.node][-1]
    OK = (ast.Subscript, ast.BinOp)
    if isinstance(parent, OK):
        return True
    if isinstance(parent, ast.Call):
        n = parent.args.index(use.node)
        return self.is_safe_call(parent.func, n)
    return False