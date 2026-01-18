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
def check_commutation(pauli_list: Sequence[PauliTerm], pauli_two: PauliTerm) -> bool:
    """
    Check if commuting a PauliTerm commutes with a list of other terms by natural calculation.
    Uses the result in Section 3 of arXiv:1405.5749v2, modified slightly here to check for the
    number of anti-coincidences (which must always be even for commuting PauliTerms)
    instead of the no. of coincidences, as in the paper.

    :param pauli_list: A list of PauliTerm objects
    :param pauli_two_term: A PauliTerm object
    :returns: True if pauli_two object commutes with pauli_list, False otherwise
    """

    def coincident_parity(p1: PauliTerm, p2: PauliTerm) -> bool:
        non_similar = 0
        p1_indices = set(p1._ops.keys())
        p2_indices = set(p2._ops.keys())
        for idx in p1_indices.intersection(p2_indices):
            if p1[idx] != p2[idx]:
                non_similar += 1
        return non_similar % 2 == 0
    for term in pauli_list:
        if not coincident_parity(term, pauli_two):
            return False
    return True