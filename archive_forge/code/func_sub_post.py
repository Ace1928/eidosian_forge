from sympy.core import Add, Basic, Mul
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.traversal import preorder_traversal
def sub_post(e):
    """ Replace 1*-1*x with -x.
    """
    replacements = []
    for node in preorder_traversal(e):
        if isinstance(node, Mul) and node.args[0] is S.One and (node.args[1] is S.NegativeOne):
            replacements.append((node, -Mul._from_args(node.args[2:])))
    for node, replacement in replacements:
        e = e.xreplace({node: replacement})
    return e