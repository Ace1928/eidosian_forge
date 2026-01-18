import operator
from functools import reduce
from itertools import chain
from nltk.sem.logic import (
def handle_refs(self):
    self.assertNextToken(DrtTokens.OPEN_BRACKET)
    refs = []
    while self.inRange(0) and self.token(0) != DrtTokens.CLOSE_BRACKET:
        if refs and self.token(0) == DrtTokens.COMMA:
            self.token()
        refs.append(self.get_next_token_variable('quantified'))
    self.assertNextToken(DrtTokens.CLOSE_BRACKET)
    return refs