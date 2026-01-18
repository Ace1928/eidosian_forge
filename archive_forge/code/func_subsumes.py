import operator
from collections import defaultdict
from functools import reduce
from nltk.inference.api import BaseProverCommand, Prover
from nltk.sem import skolemize
from nltk.sem.logic import (
def subsumes(self, other):
    """
        Return True iff 'self' subsumes 'other', this is, if there is a
        substitution such that every term in 'self' can be unified with a term
        in 'other'.

        :param other: ``Clause``
        :return: bool
        """
    negatedother = []
    for atom in other:
        if isinstance(atom, NegatedExpression):
            negatedother.append(atom.term)
        else:
            negatedother.append(-atom)
    negatedotherClause = Clause(negatedother)
    bindings = BindingDict()
    used = ([], [])
    skipped = ([], [])
    debug = DebugObject(False)
    return len(_iterate_first(self, negatedotherClause, bindings, used, skipped, _subsumes_finalize, debug)) > 0