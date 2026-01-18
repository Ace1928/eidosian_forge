from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple
from numpy.typing import NDArray
import cirq
import numpy as np
from cirq._compat import cached_property
from cirq_ft.infra import gate_with_registers, t_complexity_protocol, merge_qubits, get_named_qubits
from cirq_ft.infra.decompose_protocol import _decompose_once_considering_known_decomposition
@cached_property
def decomposed_circuit(self) -> cirq.Circuit:
    """The `gate` applied to example qubits, decomposed and wrapped in a `cirq.Circuit`."""
    return cirq.Circuit(cirq.decompose(self.operation, context=self.context))