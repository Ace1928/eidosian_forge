from typing import Optional, Sequence, Tuple
import cirq
import cirq_ft
import numpy as np
import pytest
from attr import frozen
from cirq_ft import infra
from cirq._compat import cached_property
from cirq_ft.algos.mean_estimation import CodeForRandomVariable, MeanEstimationOperator
from cirq_ft.infra import bit_tools
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@cached_property
def target_registers(self) -> Tuple[cirq_ft.Register, ...]:
    return (cirq_ft.Register('target', self.marked_val.bit_length()),)