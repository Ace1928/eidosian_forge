import abc
from typing import Sequence, Tuple
from numpy.typing import NDArray
import cirq
import numpy as np
from cirq._compat import cached_method, cached_property
from cirq_ft import infra
from cirq_ft.algos import qrom
from cirq_ft.infra.bit_tools import iter_bits
@cached_method
def rotation_gate(self, exponent: int=-1) -> cirq.Gate:
    """Returns `self._rotation_gate` ** 1 / (2 ** (1 + power))`"""
    power = 1 / 2 ** (1 + exponent)
    return cirq.pow(self._rotation_gate, power)