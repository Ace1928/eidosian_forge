from antlr4.Recognizer import Recognizer
from antlr4.RuleContext import RuleContext
from io import StringIO
def filterPrecedencePredicates(collection: set):
    return [context for context in collection if isinstance(context, PrecedencePredicate)]