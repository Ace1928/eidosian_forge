from pyparsing import *
def do_grouped_sequence(str, loc, toks):
    return Group(toks[0])