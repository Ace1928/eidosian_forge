from typing import Tuple
from numpy.typing import NDArray
import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft import infra
from cirq._compat import cached_property
from cirq_ft.infra.bit_tools import iter_bits
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@cached_property
def interleaved_unitary_target(self) -> Tuple[cirq_ft.Register, ...]:
    return tuple(cirq_ft.Signature.build(unrelated_target=1))