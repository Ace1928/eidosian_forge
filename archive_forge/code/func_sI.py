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
def sI(q: Optional[int]=None) -> PauliTerm:
    """
    A function that returns the identity operator, optionally on a particular qubit.

    This can be specified without a qubit.

    :param qubit_index: The optional index of a qubit.
    :returns: A PauliTerm object
    """
    return PauliTerm('I', q)