import ast
from .qt import ClassFlag, qt_class_flags
def format_function_def_arguments(function_def_node):
    """Formats arguments of a function definition"""
    argument_count = len(function_def_node.args.args)
    default_values = function_def_node.args.defaults
    while len(default_values) < argument_count:
        default_values.insert(0, None)
    result = ''
    for i, a in enumerate(function_def_node.args.args):
        if result:
            result += ', '
        if a.arg != 'self':
            if a.annotation and isinstance(a.annotation, ast.Name):
                result += _fix_function_argument_type(a.annotation.id, False) + ' '
            result += a.arg
            if default_values[i]:
                result += ' = '
                default_value = default_values[i]
                if isinstance(default_value, ast.Attribute):
                    result += format_reference(default_value)
                else:
                    result += format_literal(default_value)
    return result