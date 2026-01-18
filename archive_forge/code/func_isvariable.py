import operator
from functools import reduce
from itertools import chain
from nltk.sem.logic import (
def isvariable(self, tok):
    return tok not in DrtTokens.TOKENS