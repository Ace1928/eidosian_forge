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
def kappa_load_target(self) -> Tuple[infra.Register, ...]:
    return (infra.Register('kappa_load_target', self.kappa),)