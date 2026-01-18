from lib2to3.fixer_util import (FromImport, Newline, is_import,
from lib2to3.pytree import Leaf, Node
from lib2to3.pygram import python_symbols as syms
from lib2to3.pygram import token
import re
def future_import(feature, node):
    """
    This seems to work
    """
    root = find_root(node)
    if does_tree_import(u'__future__', feature, node):
        return
    shebang_encoding_idx = None
    for idx, node in enumerate(root.children):
        if is_shebang_comment(node) or is_encoding_comment(node):
            shebang_encoding_idx = idx
        if is_docstring(node):
            continue
        names = check_future_import(node)
        if not names:
            break
        if feature in names:
            return
    import_ = FromImport(u'__future__', [Leaf(token.NAME, feature, prefix=' ')])
    if shebang_encoding_idx == 0 and idx == 0:
        import_.prefix = root.children[0].prefix
        root.children[0].prefix = u''
    children = [import_, Newline()]
    root.insert_child(idx, Node(syms.simple_stmt, children))