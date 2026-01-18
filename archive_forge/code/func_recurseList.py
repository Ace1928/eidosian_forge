from pyparsing import (Literal, oneOf, printables, ParserElement, Combine,
def recurseList(elist):
    if len(elist) == 1:
        for s in elist[0].makeGenerator()():
            yield s
    else:
        for s in elist[0].makeGenerator()():
            for s2 in recurseList(elist[1:]):
                yield (s + s2)