from antlr4.Recognizer import Recognizer
from antlr4.RuleContext import RuleContext
from io import StringIO
def orContext(a: SemanticContext, b: SemanticContext):
    if a is None:
        return b
    if b is None:
        return a
    if a is SemanticContext.NONE or b is SemanticContext.NONE:
        return SemanticContext.NONE
    result = OR(a, b)
    if len(result.opnds) == 1:
        return result.opnds[0]
    else:
        return result