from pythran.passmanager import Transformation
from pythran.analyses.ast_matcher import ASTMatcher, AST_any
from pythran.conversion import mangle
from pythran.utils import isnum
import gast as ast
import copy
def isNone(self, node):
    if node is None:
        return True
    return isinstance(node, ast.Constant) and node.value is None