from pyparsing import (Literal, oneOf, printables, ParserElement, Combine,
def altGen():
    for e in self.exprs:
        for s in e.makeGenerator()():
            yield s