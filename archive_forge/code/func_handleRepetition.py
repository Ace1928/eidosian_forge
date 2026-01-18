from pyparsing import (Literal, oneOf, printables, ParserElement, Combine,
def handleRepetition(toks):
    toks = toks[0]
    if toks[1] in '*+':
        raise ParseFatalException('', 0, 'unbounded repetition operators not supported')
    if toks[1] == '?':
        return OptionalEmitter(toks[0])
    if 'count' in toks:
        return GroupEmitter([toks[0]] * int(toks.count))
    if 'minCount' in toks:
        mincount = int(toks.minCount)
        maxcount = int(toks.maxCount)
        optcount = maxcount - mincount
        if optcount:
            opt = OptionalEmitter(toks[0])
            for i in range(1, optcount):
                opt = OptionalEmitter(GroupEmitter([toks[0], opt]))
            return GroupEmitter([toks[0]] * mincount + [opt])
        else:
            return [toks[0]] * mincount