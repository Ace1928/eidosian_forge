from pyparsing import (Literal, oneOf, printables, ParserElement, Combine,
def handleRange(toks):
    return CharacterRangeEmitter(srange(toks[0]))