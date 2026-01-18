from pythran.passmanager import FunctionAnalysis
from pythran.utils import isnum
from pythran.graph import DiGraph
import gast as ast
def is_true_predicate(node):
    if isnum(node) and node.value:
        return True
    if isinstance(node, ast.Attribute) and node.attr == 'True':
        return True
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)) and node.elts:
        return True
    if isinstance(node, ast.Dict) and node.keys:
        return True
    return False