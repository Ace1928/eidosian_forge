import operator
from functools import reduce
from itertools import chain
from nltk.sem.logic import (
def eliminate_equality(self):
    drs = self.simplify()
    assert not isinstance(drs, DrtConcatenation)
    return drs.eliminate_equality()