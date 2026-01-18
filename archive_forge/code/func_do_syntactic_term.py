from pyparsing import *
def do_syntactic_term(str, loc, toks):
    if len(toks) == 2:
        return NotAny(toks[1]) + toks[0]
    else:
        return [toks[0]]