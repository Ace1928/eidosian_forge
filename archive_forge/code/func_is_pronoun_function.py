import operator
from functools import reduce
from itertools import chain
from nltk.sem.logic import (
def is_pronoun_function(self):
    """Is self of the form "PRO(x)"?"""
    return isinstance(self, DrtApplicationExpression) and isinstance(self.function, DrtAbstractVariableExpression) and (self.function.variable.name == DrtTokens.PRONOUN) and isinstance(self.argument, DrtIndividualVariableExpression)