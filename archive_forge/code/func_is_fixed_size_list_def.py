from pythran.passmanager import FunctionAnalysis
from pythran.tables import MODULES
import gast as ast
def is_fixed_size_list_def(self, node):
    if isinstance(node, ast.List):
        return True
    if not isinstance(node, ast.Call):
        return False
    return all((alias == MODULES['builtins']['list'] for alias in self.aliases[node.func]))