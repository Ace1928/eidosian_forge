from pyparsing import (
import pydot
def push_attr_list(s, loc, toks):
    p = P_AttrList(toks)
    return p