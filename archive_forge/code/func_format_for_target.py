import ast
from .qt import ClassFlag, qt_class_flags
def format_for_target(target_node):
    if isinstance(target_node, ast.Tuple):
        result = ''
        for i, el in enumerate(target_node.elts):
            if i > 0:
                result += ', '
            result += format_reference(el)
        return result
    return format_reference(target_node)