import re
import sys
def format_lines(var_name, seq, raw=False, indent_level=0):
    """Formats a sequence of strings for output."""
    lines = []
    base_indent = ' ' * indent_level * 4
    inner_indent = ' ' * (indent_level + 1) * 4
    lines.append(base_indent + var_name + ' = (')
    if raw:
        for i in seq:
            lines.append(inner_indent + i + ',')
    else:
        for i in seq:
            r = repr(i + '"')
            lines.append(inner_indent + r[:-2] + r[-1] + ',')
    lines.append(base_indent + ')')
    return '\n'.join(lines)