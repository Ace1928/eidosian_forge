from pyparsing import (Literal, oneOf, printables, ParserElement, Combine,
def handleAlternative(toks):
    return AlternativeEmitter(toks[0])