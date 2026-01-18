from pyparsing import (Literal, oneOf, printables, ParserElement, Combine,
def handleDot():
    return CharacterRangeEmitter(printables)