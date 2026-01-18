import ast
from .qt import ClassFlag, qt_class_flags
def format_literal(node):
    """Returns the value of number/string literals"""
    if isinstance(node, ast.NameConstant):
        return format_name_constant(node)
    if isinstance(node, ast.Num):
        return str(node.n)
    if isinstance(node, ast.Str):
        return f'"{node.s}"'
    return ''