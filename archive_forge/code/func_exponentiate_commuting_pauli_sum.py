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
def exponentiate_commuting_pauli_sum(pauli_sum: PauliSum) -> Callable[[Union[float, MemoryReference]], Program]:
    """
    Returns a function that maps all substituent PauliTerms and sums them into a program. NOTE: Use
    this function with care. Substituent PauliTerms should commute.

    :param pauli_sum: PauliSum to exponentiate.
    :returns: A function that parametrizes the exponential.
    """
    if not isinstance(pauli_sum, PauliSum):
        raise TypeError("Argument 'pauli_sum' must be a PauliSum.")
    fns = [exponential_map(term) for term in pauli_sum]

    def combined_exp_wrap(param: Union[float, MemoryReference]) -> Program:
        return Program([f(param) for f in fns])
    return combined_exp_wrap