import operator
from functools import reduce
from itertools import chain
from nltk.sem.logic import (
def handle_DRS(self, tok, context):
    refs = self.handle_refs()
    if self.inRange(0) and self.token(0) == DrtTokens.COMMA:
        self.token()
    conds = self.handle_conds(context)
    self.assertNextToken(DrtTokens.CLOSE)
    return DRS(refs, conds, None)