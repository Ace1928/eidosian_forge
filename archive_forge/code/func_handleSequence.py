from pyparsing import (Literal, oneOf, printables, ParserElement, Combine,
def handleSequence(toks):
    return GroupEmitter(toks[0])