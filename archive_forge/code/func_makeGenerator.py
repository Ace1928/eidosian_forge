from pyparsing import (Literal, oneOf, printables, ParserElement, Combine,
def makeGenerator(self):

    def litGen():
        yield self.lit
    return litGen