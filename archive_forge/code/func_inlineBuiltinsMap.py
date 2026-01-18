from pythran.analyses import Aliases
from pythran.passmanager import Transformation
from pythran.tables import MODULES
from pythran.intrinsic import FunctionIntr
from pythran.utils import path_to_attr, path_to_node
from pythran.syntax import PythranSyntaxError
from copy import deepcopy
import gast as ast
def inlineBuiltinsMap(self, node):
    if not isinstance(node, ast.Call):
        return node
    func_aliases = self.aliases[node.func]
    if len(func_aliases) != 1:
        return node
    obj = next(iter(func_aliases))
    if obj is not MODULES['builtins']['map']:
        return node
    if not all((isinstance(arg, (ast.List, ast.Tuple)) for arg in node.args[1:])):
        return node
    mapped_func_aliases = self.aliases[node.args[0]]
    if len(mapped_func_aliases) != 1:
        return node
    obj = next(iter(mapped_func_aliases))
    if not isinstance(obj, (ast.FunctionDef, FunctionIntr)):
        return node
    return self.inlineBuiltinsXMap(node)