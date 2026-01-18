from lib2to3 import fixer_base
from lib2to3.pygram import token
from lib2to3.fixer_util import Name, syms, Node, Leaf, touch_import, Call, \
def find_metas(cls_node):
    for node in cls_node.children:
        if node.type == syms.suite:
            break
    else:
        raise ValueError('No class suite!')
    for i, simple_node in list(enumerate(node.children)):
        if simple_node.type == syms.simple_stmt and simple_node.children:
            expr_node = simple_node.children[0]
            if expr_node.type == syms.expr_stmt and expr_node.children:
                left_node = expr_node.children[0]
                if isinstance(left_node, Leaf) and left_node.value == u'__metaclass__':
                    fixup_simple_stmt(node, i, simple_node)
                    remove_trailing_newline(simple_node)
                    yield (node, i, simple_node)