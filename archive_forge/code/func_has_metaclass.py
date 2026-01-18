from lib2to3 import fixer_base
from lib2to3.pygram import token
from lib2to3.fixer_util import Name, syms, Node, Leaf, touch_import, Call, \
def has_metaclass(parent):
    """ we have to check the cls_node without changing it.
        There are two possibilities:
          1)  clsdef => suite => simple_stmt => expr_stmt => Leaf('__meta')
          2)  clsdef => simple_stmt => expr_stmt => Leaf('__meta')
    """
    for node in parent.children:
        if node.type == syms.suite:
            return has_metaclass(node)
        elif node.type == syms.simple_stmt and node.children:
            expr_node = node.children[0]
            if expr_node.type == syms.expr_stmt and expr_node.children:
                left_side = expr_node.children[0]
                if isinstance(left_side, Leaf) and left_side.value == '__metaclass__':
                    return True
    return False