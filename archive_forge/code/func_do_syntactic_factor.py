from pyparsing import *
def do_syntactic_factor(str, loc, toks):
    if len(toks) == 2:
        return And([toks[1]] * toks[0])
    else:
        return [toks[0]]