from pyparsing import *
def do_optional_sequence(str, loc, toks):
    return Optional(toks[0])