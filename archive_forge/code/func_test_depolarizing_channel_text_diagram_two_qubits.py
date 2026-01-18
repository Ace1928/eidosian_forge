import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_depolarizing_channel_text_diagram_two_qubits():
    d = cirq.depolarize(0.1234567, n_qubits=2)
    assert cirq.circuit_diagram_info(d, args=round_to_6_prec) == cirq.CircuitDiagramInfo(wire_symbols=('D(0.123457)', '#2'))
    assert cirq.circuit_diagram_info(d, args=round_to_2_prec) == cirq.CircuitDiagramInfo(wire_symbols=('D(0.12)', '#2'))
    assert cirq.circuit_diagram_info(d, args=no_precision) == cirq.CircuitDiagramInfo(wire_symbols=('D(0.1234567)', '#2'))