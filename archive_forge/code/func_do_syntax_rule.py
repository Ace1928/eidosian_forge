from pyparsing import *
def do_syntax_rule(str, loc, toks):
    assert toks[0].expr is None, 'Duplicate definition'
    forward_count.value -= 1
    toks[0] << toks[1]
    return [toks[0]]