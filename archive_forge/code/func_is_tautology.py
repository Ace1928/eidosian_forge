import operator
from collections import defaultdict
from functools import reduce
from nltk.inference.api import BaseProverCommand, Prover
from nltk.sem import skolemize
from nltk.sem.logic import (
def is_tautology(self):
    """
        Self is a tautology if it contains ground terms P and -P.  The ground
        term, P, must be an exact match, ie, not using unification.
        """
    if self._is_tautology is not None:
        return self._is_tautology
    for i, a in enumerate(self):
        if not isinstance(a, EqualityExpression):
            j = len(self) - 1
            while j > i:
                b = self[j]
                if isinstance(a, NegatedExpression):
                    if a.term == b:
                        self._is_tautology = True
                        return True
                elif isinstance(b, NegatedExpression):
                    if a == b.term:
                        self._is_tautology = True
                        return True
                j -= 1
    self._is_tautology = False
    return False