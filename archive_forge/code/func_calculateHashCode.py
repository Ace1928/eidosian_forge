from io import StringIO
from antlr4.error.Errors import IllegalStateException
from antlr4.RuleContext import RuleContext
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNState import ATNState
def calculateHashCode(parent: PredictionContext, returnState: int):
    return hash('') if parent is None else hash((hash(parent), returnState))