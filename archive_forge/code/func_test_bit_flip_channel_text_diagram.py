import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_bit_flip_channel_text_diagram():
    bf = cirq.bit_flip(0.1234567)
    assert cirq.circuit_diagram_info(bf, args=round_to_6_prec) == cirq.CircuitDiagramInfo(wire_symbols=('BF(0.123457)',))
    assert cirq.circuit_diagram_info(bf, args=round_to_2_prec) == cirq.CircuitDiagramInfo(wire_symbols=('BF(0.12)',))
    assert cirq.circuit_diagram_info(bf, args=no_precision) == cirq.CircuitDiagramInfo(wire_symbols=('BF(0.1234567)',))