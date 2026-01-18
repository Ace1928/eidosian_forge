import pyglet.gl as pgl
from sympy.core import S
def parse_option_string(s):
    if not isinstance(s, str):
        return None
    options = {}
    for token in s.split(';'):
        pieces = token.split('=')
        if len(pieces) == 1:
            option, value = (pieces[0], '')
        elif len(pieces) == 2:
            option, value = pieces
        else:
            raise ValueError("Plot option string '%s' is malformed." % s)
        options[option.strip()] = value.strip()
    return options