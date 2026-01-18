from lib2to3 import fixer_base
from lib2to3.pygram import token
from lib2to3.fixer_util import Name, syms, Node, Leaf, touch_import, Call, \
def fixup_simple_stmt(parent, i, stmt_node):
    """ if there is a semi-colon all the parts count as part of the same
        simple_stmt.  We just want the __metaclass__ part so we move
        everything efter the semi-colon into its own simple_stmt node
    """
    for semi_ind, node in enumerate(stmt_node.children):
        if node.type == token.SEMI:
            break
    else:
        return
    node.remove()
    new_expr = Node(syms.expr_stmt, [])
    new_stmt = Node(syms.simple_stmt, [new_expr])
    while stmt_node.children[semi_ind:]:
        move_node = stmt_node.children[semi_ind]
        new_expr.append_child(move_node.clone())
        move_node.remove()
    parent.insert_child(i, new_stmt)
    new_leaf1 = new_stmt.children[0].children[0]
    old_leaf1 = stmt_node.children[0].children[0]
    new_leaf1.prefix = old_leaf1.prefix