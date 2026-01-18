from typing import List, Tuple
from numpy.typing import NDArray
import attr
import cirq
import numpy as np
from cirq._compat import cached_property
from cirq_ft import infra, linalg
from cirq_ft.algos import (
@cached_property
def selection_bitsize(self) -> int:
    return infra.total_bits(self.selection_registers)