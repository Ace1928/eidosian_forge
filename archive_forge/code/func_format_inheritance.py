import ast
from .qt import ClassFlag, qt_class_flags
def format_inheritance(class_def_node):
    """Returns inheritance specification of a class"""
    result = ''
    for base in class_def_node.bases:
        name = to_string(base)
        if name != 'object':
            result += ', public ' if result else ' : public '
            result += name
    return result