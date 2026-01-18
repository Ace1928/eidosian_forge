import operator
from collections import defaultdict
from functools import reduce
from nltk.inference.api import BaseProverCommand, Prover
from nltk.sem import skolemize
from nltk.sem.logic import (
def isSubsetOf(self, other):
    """
        Return True iff every term in 'self' is a term in 'other'.

        :param other: ``Clause``
        :return: bool
        """
    for a in self:
        if a not in other:
            return False
    return True