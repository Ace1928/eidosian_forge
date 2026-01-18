from lib2to3.fixer_util import (FromImport, Newline, is_import,
from lib2to3.pytree import Leaf, Node
from lib2to3.pygram import python_symbols as syms
from lib2to3.pygram import token
import re
def is_docstring(node):
    """
    Returns True if the node appears to be a docstring
    """
    return node.type == syms.simple_stmt and len(node.children) > 0 and (node.children[0].type == token.STRING)