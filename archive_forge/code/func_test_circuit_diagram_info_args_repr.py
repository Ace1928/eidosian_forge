import numpy as np
import pytest
import sympy
import cirq
def test_circuit_diagram_info_args_repr():
    cirq.testing.assert_equivalent_repr(cirq.CircuitDiagramInfoArgs(known_qubits=cirq.LineQubit.range(2), known_qubit_count=2, use_unicode_characters=True, precision=5, label_map={cirq.LineQubit(0): 5, cirq.LineQubit(1): 7}, include_tags=False, transpose=True))