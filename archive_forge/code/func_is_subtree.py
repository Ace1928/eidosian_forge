from lib2to3.pgen2 import token
from lib2to3.pygram import python_symbols as syms
from lib2to3 import fixer_base
from lib2to3.fixer_util import Name, Call, find_binding
def is_subtree(root, node):
    if root == node:
        return True
    return any((is_subtree(c, node) for c in root.children))