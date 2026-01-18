from lib2to3.fixer_util import (FromImport, Newline, is_import,
from lib2to3.pytree import Leaf, Node
from lib2to3.pygram import python_symbols as syms
from lib2to3.pygram import token
import re
def future_import2(feature, node):
    """
    An alternative to future_import() which might not work ...
    """
    root = find_root(node)
    if does_tree_import(u'__future__', feature, node):
        return
    insert_pos = 0
    for idx, node in enumerate(root.children):
        if node.type == syms.simple_stmt and node.children and (node.children[0].type == token.STRING):
            insert_pos = idx + 1
            break
    for thing_after in root.children[insert_pos:]:
        if thing_after.type == token.NEWLINE:
            insert_pos += 1
            continue
        prefix = thing_after.prefix
        thing_after.prefix = u''
        break
    else:
        prefix = u''
    import_ = FromImport(u'__future__', [Leaf(token.NAME, feature, prefix=u' ')])
    children = [import_, Newline()]
    root.insert_child(insert_pos, Node(syms.simple_stmt, children, prefix=prefix))