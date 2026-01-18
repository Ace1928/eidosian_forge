import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_generalized_amplitude_damping_channel_text_diagram():
    a = cirq.generalized_amplitude_damp(0.1, 0.39558391)
    assert cirq.circuit_diagram_info(a, args=round_to_6_prec) == cirq.CircuitDiagramInfo(wire_symbols=('GAD(0.1,0.395584)',))
    assert cirq.circuit_diagram_info(a, args=round_to_2_prec) == cirq.CircuitDiagramInfo(wire_symbols=('GAD(0.1,0.4)',))
    assert cirq.circuit_diagram_info(a, args=no_precision) == cirq.CircuitDiagramInfo(wire_symbols=('GAD(0.1,0.39558391)',))