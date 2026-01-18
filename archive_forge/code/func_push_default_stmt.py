from pyparsing import (
import pydot
def push_default_stmt(s, loc, toks):
    default_type = toks[0][0]
    if len(toks) > 1:
        attrs = toks[1].attrs
    else:
        attrs = {}
    if default_type in ['graph', 'node', 'edge']:
        return DefaultStatement(default_type, attrs)
    else:
        raise ValueError('Unknown default statement: {s}'.format(s=toks))