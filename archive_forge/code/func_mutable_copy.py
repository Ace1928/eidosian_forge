import cmath
import math
import numbers
from typing import (
import numpy as np
import sympy
import cirq
from cirq import value, protocols, linalg, qis
from cirq._doc import document
from cirq._import import LazyLoader
from cirq.ops import (
from cirq.type_workarounds import NotImplementedType
def mutable_copy(self) -> 'cirq.MutablePauliString':
    """Returns a new `cirq.MutablePauliString` with the same contents."""
    return MutablePauliString(coefficient=self.coefficient, pauli_int_dict=dict(self.pauli_int_dict))