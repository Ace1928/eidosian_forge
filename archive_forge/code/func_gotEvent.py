from pyparsing import Optional, oneOf, Literal, Word, printables, Group, OneOrMore, ZeroOrMore
def gotEvent(s, loc, toks):
    for event in toks:
        print(event.dump())