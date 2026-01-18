from io import StringIO
from antlr4.error.Errors import IllegalStateException
from antlr4.RuleContext import RuleContext
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNState import ATNState
def mergeRoot(a: SingletonPredictionContext, b: SingletonPredictionContext, rootIsWildcard: bool):
    if rootIsWildcard:
        if a == PredictionContext.EMPTY:
            return PredictionContext.EMPTY
        if b == PredictionContext.EMPTY:
            return PredictionContext.EMPTY
    elif a == PredictionContext.EMPTY and b == PredictionContext.EMPTY:
        return PredictionContext.EMPTY
    elif a == PredictionContext.EMPTY:
        payloads = [b.returnState, PredictionContext.EMPTY_RETURN_STATE]
        parents = [b.parentCtx, None]
        return ArrayPredictionContext(parents, payloads)
    elif b == PredictionContext.EMPTY:
        payloads = [a.returnState, PredictionContext.EMPTY_RETURN_STATE]
        parents = [a.parentCtx, None]
        return ArrayPredictionContext(parents, payloads)
    return None