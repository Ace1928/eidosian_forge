from pyparsing import *
def createTokenObject(st, locn, toks):
    return Token(st, locn, toks[0])