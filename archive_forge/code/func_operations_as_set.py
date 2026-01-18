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
def operations_as_set(self) -> FrozenSet[Tuple[PauliTargetDesignator, str]]:
    """
        Return a frozenset of operations in this term.

        Use this in place of :py:func:`id` if the order of operations in the term does not
        matter.

        :return: frozenset of (qubit, op_str) representing Pauli operations
        """
    return frozenset(self._ops.items())