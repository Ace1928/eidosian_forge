import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_amplitude_damping_channel_text_diagram():
    ad = cirq.amplitude_damp(0.38059322)
    assert cirq.circuit_diagram_info(ad, args=round_to_6_prec) == cirq.CircuitDiagramInfo(wire_symbols=('AD(0.380593)',))
    assert cirq.circuit_diagram_info(ad, args=round_to_2_prec) == cirq.CircuitDiagramInfo(wire_symbols=('AD(0.38)',))
    assert cirq.circuit_diagram_info(ad, args=no_precision) == cirq.CircuitDiagramInfo(wire_symbols=('AD(0.38059322)',))