from lib2to3.fixer_util import (FromImport, Newline, is_import,
from lib2to3.pytree import Leaf, Node
from lib2to3.pygram import python_symbols as syms
from lib2to3.pygram import token
import re
def check_future_import(node):
    """If this is a future import, return set of symbols that are imported,
    else return None."""
    savenode = node
    if not (node.type == syms.simple_stmt and node.children):
        return set()
    node = node.children[0]
    if not (node.type == syms.import_from and hasattr(node.children[1], 'value') and (node.children[1].value == u'__future__')):
        return set()
    if node.children[3].type == token.LPAR:
        node = node.children[4]
    else:
        node = node.children[3]
    if node.type == syms.import_as_names:
        result = set()
        for n in node.children:
            if n.type == token.NAME:
                result.add(n.value)
            elif n.type == syms.import_as_name:
                n = n.children[0]
                assert n.type == token.NAME
                result.add(n.value)
        return result
    elif node.type == syms.import_as_name:
        node = node.children[0]
        assert node.type == token.NAME
        return set([node.value])
    elif node.type == token.NAME:
        return set([node.value])
    else:
        assert False, 'strange import: %s' % savenode