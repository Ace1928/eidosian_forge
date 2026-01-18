import re
def escape_quotes(m):
    """Used in a regex to properly escape double quotes."""
    s = m[0]
    if s == '"':
        return '\\"'
    else:
        return s