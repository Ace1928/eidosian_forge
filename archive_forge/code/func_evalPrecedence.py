from antlr4.Recognizer import Recognizer
from antlr4.RuleContext import RuleContext
from io import StringIO
def evalPrecedence(self, parser: Recognizer, outerContext: RuleContext):
    differs = False
    operands = []
    for context in self.opnds:
        evaluated = context.evalPrecedence(parser, outerContext)
        differs |= evaluated is not context
        if evaluated is SemanticContext.NONE:
            return SemanticContext.NONE
        elif evaluated is not None:
            operands.append(evaluated)
    if not differs:
        return self
    if len(operands) == 0:
        return None
    result = None
    for o in operands:
        result = o if result is None else orContext(result, o)
    return result