from pythran.passmanager import Transformation
from pythran.analyses.ast_matcher import ASTMatcher, AST_any
from pythran.conversion import mangle
from pythran.utils import isnum
import gast as ast
import copy
def is_full_slice(self, node):
    return isinstance(node, ast.Slice) and (node.lower == 0 or self.isNone(node.lower)) and self.isNone(node.upper) and (self.isNone(node.step) or node.step == 1)