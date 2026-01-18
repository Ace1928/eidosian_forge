from typing import Sequence
import numpy as np
from pyquil.gates import MEASURE, RX, RZ
from pyquil.quil import Program
def measure_qubits(qubits: Sequence[int]) -> Program:
    """
    Given a number of qubits (n), produce a ``Program`` with a ``MEASURE`` instruction on qubits
    0 through n-1, with corresponding readout registers ro[0] through ro[n-1].

    :param qubits: The number of qubits (n).
    :return: A ``Program`` that measures n qubits.
    """
    p = Program()
    ro = p.declare('ro', 'BIT', len(qubits))
    for idx, q in enumerate(qubits):
        p += MEASURE(q, ro[idx])
    return p