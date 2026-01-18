import operator
from functools import reduce
from itertools import chain
from nltk.sem.logic import (
def make_BooleanExpression(self, factory, first, second):
    return factory(first, second)