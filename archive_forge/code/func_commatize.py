from lib2to3.fixer_util import (FromImport, Newline, is_import,
from lib2to3.pytree import Leaf, Node
from lib2to3.pygram import python_symbols as syms
from lib2to3.pygram import token
import re
def commatize(leafs):
    """
    Accepts/turns: (Name, Name, ..., Name, Name)
    Returns/into: (Name, Comma, Name, Comma, ..., Name, Comma, Name)
    """
    new_leafs = []
    for leaf in leafs:
        new_leafs.append(leaf)
        new_leafs.append(Comma())
    del new_leafs[-1]
    return new_leafs