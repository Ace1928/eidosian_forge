from io import StringIO
from antlr4.error.Errors import IllegalStateException
from antlr4.RuleContext import RuleContext
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNState import ATNState
def calculateListsHashCode(parents: [], returnStates: []):
    h = 0
    for parent, returnState in zip(parents, returnStates):
        h = hash((h, calculateHashCode(parent, returnState)))
    return h