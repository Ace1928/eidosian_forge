from pyparsing import *
def do_terminal_string(str, loc, toks):
    return Literal(toks[0])