import itertools
import os
import time
from collections import defaultdict
from random import randint, random, sample, randrange
from typing import Iterator, Optional, Tuple, TYPE_CHECKING
import numpy as np
import pytest
import sympy
import cirq
from cirq import circuits
from cirq import ops
from cirq.testing.devices import ValidatingTestDevice
def pauli_error_probability(r: float, n_qubits: int) -> float:
    """Computes Pauli error probability for given depolarization parameter.

        Pauli error is what cirq.depolarize takes as argument. Depolarization parameter
        makes it simple to compute the serial composition of depolarizing channels. It
        is multiplicative under channel composition.
        """
    d2 = 4 ** n_qubits
    return (1 - r) * (d2 - 1) / d2