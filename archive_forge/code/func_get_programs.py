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
def get_programs(self) -> Tuple[List[Program], np.ndarray]:
    """
        Get a Pyquil Program corresponding to each term in the PauliSum and a coefficient
        for each program

        :return: (programs, coefficients)
        """
    programs = [term.program for term in self.terms]
    coefficients = np.array([term.coefficient for term in self.terms])
    return (programs, coefficients)