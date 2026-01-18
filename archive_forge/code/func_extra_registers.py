from typing import Sequence, Union, Tuple
from numpy.typing import NDArray
import attr
import cirq
import numpy as np
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.algos import unary_iteration_gate
@cached_property
def extra_registers(self) -> Tuple[infra.Register, ...]:
    return (infra.Register('accumulator', 1),)