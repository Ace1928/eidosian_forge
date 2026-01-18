import re
from itertools import product
import numpy as np
import copy
from typing import (
from pyquil.quilatom import (
from .quil import Program
from .gates import H, RZ, RX, CNOT, X, PHASE, QUANTUM_GATES
from numbers import Number, Complex
from collections import OrderedDict
import warnings
def simplify_pauli_sum(pauli_sum: PauliSum) -> PauliSum:
    """
    Simplify the sum of Pauli operators according to Pauli algebra rules.

    Warning: The simplified expression may re-order pauli operations, and may
    impact the observed performance when running on the QPU.
    """
    like_terms: Dict[Hashable, List[PauliTerm]] = OrderedDict()
    for term in pauli_sum.terms:
        key = term.operations_as_set()
        if key in like_terms:
            like_terms[key].append(term)
        else:
            like_terms[key] = [term]
    terms = []
    for term_list in like_terms.values():
        first_term = term_list[0]
        if len(term_list) == 1 and (not np.isclose(first_term.coefficient, 0.0)):
            terms.append(first_term)
        else:
            coeff = sum((t.coefficient for t in term_list))
            if not np.isclose(coeff, 0.0):
                terms.append(term_with_coeff(term_list[0], coeff))
    return PauliSum(terms)