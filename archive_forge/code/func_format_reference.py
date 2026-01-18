import ast
from .qt import ClassFlag, qt_class_flags
def format_reference(node, qualifier='auto'):
    """Format member reference or free item"""
    return node.id if isinstance(node, ast.Name) else format_member(node, qualifier)