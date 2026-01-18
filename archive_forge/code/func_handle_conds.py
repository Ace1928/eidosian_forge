import operator
from functools import reduce
from itertools import chain
from nltk.sem.logic import (
def handle_conds(self, context):
    self.assertNextToken(DrtTokens.OPEN_BRACKET)
    conds = []
    while self.inRange(0) and self.token(0) != DrtTokens.CLOSE_BRACKET:
        if conds and self.token(0) == DrtTokens.COMMA:
            self.token()
        conds.append(self.process_next_expression(context))
    self.assertNextToken(DrtTokens.CLOSE_BRACKET)
    return conds