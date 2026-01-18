import abc
from typing import Sequence, Tuple
from numpy.typing import NDArray
import cirq
import numpy as np
from cirq._compat import cached_method, cached_property
from cirq_ft import infra
from cirq_ft.algos import qrom
from cirq_ft.infra.bit_tools import iter_bits
@cached_property
def rotations_target(self) -> Tuple[infra.Register, ...]:
    return (infra.Register('rotations_target', self._target_bitsize),)