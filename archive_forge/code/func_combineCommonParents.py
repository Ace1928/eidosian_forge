from io import StringIO
from antlr4.error.Errors import IllegalStateException
from antlr4.RuleContext import RuleContext
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNState import ATNState
def combineCommonParents(parents: list):
    uniqueParents = dict()
    for p in range(0, len(parents)):
        parent = parents[p]
        if uniqueParents.get(parent, None) is None:
            uniqueParents[parent] = parent
    for p in range(0, len(parents)):
        parents[p] = uniqueParents[parents[p]]