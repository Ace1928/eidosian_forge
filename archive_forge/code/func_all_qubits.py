from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple
from numpy.typing import NDArray
import cirq
import numpy as np
from cirq._compat import cached_property
from cirq_ft.infra import gate_with_registers, t_complexity_protocol, merge_qubits, get_named_qubits
from cirq_ft.infra.decompose_protocol import _decompose_once_considering_known_decomposition
@cached_property
def all_qubits(self) -> List[cirq.Qid]:
    """All qubits in Register order."""
    merged_qubits = merge_qubits(self.r, **self.quregs)
    decomposed_qubits = self.decomposed_circuit.all_qubits()
    return merged_qubits + sorted(decomposed_qubits - frozenset(merged_qubits))