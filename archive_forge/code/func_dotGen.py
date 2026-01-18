from pyparsing import (Literal, oneOf, printables, ParserElement, Combine,
def dotGen():
    for c in printables:
        yield c