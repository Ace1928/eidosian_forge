from antlr4.PredictionContext import PredictionContextCache, PredictionContext, getCachedPredictionContext
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNConfigSet import ATNConfigSet
from antlr4.dfa.DFAState import DFAState
def getCachedContext(self, context: PredictionContext):
    if self.sharedContextCache is None:
        return context
    visited = dict()
    return getCachedPredictionContext(context, self.sharedContextCache, visited)