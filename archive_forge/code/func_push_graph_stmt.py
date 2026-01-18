from pyparsing import (
import pydot
def push_graph_stmt(s, loc, toks):
    g = pydot.Subgraph('')
    add_elements(g, toks)
    return g