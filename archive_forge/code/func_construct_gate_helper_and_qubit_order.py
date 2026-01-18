import itertools
import cirq
import cirq_ft
from cirq_ft import infra
import numpy as np
import pytest
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
def construct_gate_helper_and_qubit_order(gate):
    g = cirq_ft.testing.GateHelper(gate)
    context = cirq.DecompositionContext(cirq.ops.SimpleQubitManager())
    circuit = cirq.Circuit(cirq.decompose(g.operation, keep=keep, on_stuck_raise=None, context=context))
    ordered_input = list(itertools.chain(*g.quregs.values()))
    qubit_order = cirq.QubitOrder.explicit(ordered_input, fallback=cirq.QubitOrder.DEFAULT)
    return (g, qubit_order, circuit)