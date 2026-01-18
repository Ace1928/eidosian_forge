import operator
from functools import reduce
from itertools import chain
from nltk.sem.logic import (
def handle_prop(self, tok, context):
    variable = self.make_VariableExpression(tok)
    self.assertNextToken(':')
    drs = self.process_next_expression(DrtTokens.COLON)
    return DrtProposition(variable, drs)