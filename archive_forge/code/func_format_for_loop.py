import ast
from .qt import ClassFlag, qt_class_flags
def format_for_loop(f_node):
    """Format a for loop
       This applies some heuristics to detect:
       1) "for a in [1,2])" -> "for (f: {1, 2}) {"
       2) "for i in range(5)" -> "for (i = 0; i < 5; ++i) {"
       3) "for i in range(2,5)" -> "for (i = 2; i < 5; ++i) {"

       TODO: Detect other cases, maybe including enumerate().
    """
    loop_vars = format_for_target(f_node.target)
    result = 'for (' + loop_vars
    if isinstance(f_node.iter, ast.Call):
        f = format_reference(f_node.iter.func)
        if f == 'range':
            start = 0
            end = -1
            if len(f_node.iter.args) == 2:
                start = format_literal(f_node.iter.args[0])
                end = format_literal(f_node.iter.args[1])
            elif len(f_node.iter.args) == 1:
                end = format_literal(f_node.iter.args[0])
            result += f' = {start}; {loop_vars} < {end}; ++{loop_vars}'
    elif isinstance(f_node.iter, ast.List):
        result += ': ' + format_literal_list(f_node.iter)
    elif isinstance(f_node.iter, ast.Name):
        result += ': ' + f_node.iter.id
    result += ') {'
    return result