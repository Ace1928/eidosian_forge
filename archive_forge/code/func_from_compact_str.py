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
@classmethod
def from_compact_str(cls, str_pauli_sum: str) -> 'PauliSum':
    """Construct a PauliSum from the result of str(pauli_sum)"""
    str_terms = re.split('\\+(?![^(]*\\))', str_pauli_sum)
    str_terms = [s.strip() for s in str_terms]
    terms = [PauliTerm.from_compact_str(term) for term in str_terms]
    return cls(terms).simplify()