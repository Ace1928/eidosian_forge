from pyparsing import *
def do_single_definition(str, loc, toks):
    toks = toks.asList()
    if len(toks) > 1:
        return And(toks)
    else:
        return [toks[0]]