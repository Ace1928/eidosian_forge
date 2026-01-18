from pyparsing import (
import pydot
def push_node_stmt(s, loc, toks):
    if len(toks) == 2:
        attrs = toks[1].attrs
    else:
        attrs = {}
    node_name = toks[0]
    if isinstance(node_name, list) or isinstance(node_name, tuple):
        if len(node_name) > 0:
            node_name = node_name[0]
    n = pydot.Node(str(node_name), **attrs)
    return n