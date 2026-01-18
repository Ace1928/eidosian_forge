from io import StringIO
from antlr4.error.Errors import IllegalStateException
from antlr4.RuleContext import RuleContext
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNState import ATNState
def getCachedPredictionContext(context: PredictionContext, contextCache: PredictionContextCache, visited: dict):
    if context.isEmpty():
        return context
    existing = visited.get(context)
    if existing is not None:
        return existing
    existing = contextCache.get(context)
    if existing is not None:
        visited[context] = existing
        return existing
    changed = False
    parents = [None] * len(context)
    for i in range(0, len(parents)):
        parent = getCachedPredictionContext(context.getParent(i), contextCache, visited)
        if changed or parent is not context.getParent(i):
            if not changed:
                parents = [context.getParent(j) for j in range(len(context))]
                changed = True
            parents[i] = parent
    if not changed:
        contextCache.add(context)
        visited[context] = context
        return context
    updated = None
    if len(parents) == 0:
        updated = PredictionContext.EMPTY
    elif len(parents) == 1:
        updated = SingletonPredictionContext.create(parents[0], context.getReturnState(0))
    else:
        updated = ArrayPredictionContext(parents, context.returnStates)
    contextCache.add(updated)
    visited[updated] = updated
    visited[context] = updated
    return updated