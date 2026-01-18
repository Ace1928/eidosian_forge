import functools
import gast
def is_literal(node):
    """Tests whether node represents a Python literal."""
    if is_constant(node):
        return True
    if isinstance(node, gast.Name) and node.id in ['True', 'False', 'None']:
        return True
    return False