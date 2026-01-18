from typing import Union, Tuple, cast
import numpy as np
import pytest
import sympy
import cirq
from cirq.type_workarounds import NotImplementedType
def test_circuit_diagram_info():
    assert cirq.circuit_diagram_info(CY) == cirq.CircuitDiagramInfo(wire_symbols=('@', 'Y'), exponent=1)
    assert cirq.circuit_diagram_info(C0Y) == cirq.CircuitDiagramInfo(wire_symbols=('(0)', 'Y'), exponent=1)
    assert cirq.circuit_diagram_info(C2Y) == cirq.CircuitDiagramInfo(wire_symbols=('(2)', 'Y'), exponent=1)
    assert cirq.circuit_diagram_info(cirq.ControlledGate(cirq.Y ** 0.5)) == cirq.CircuitDiagramInfo(wire_symbols=('@', 'Y'), exponent=0.5)
    assert cirq.circuit_diagram_info(cirq.ControlledGate(cirq.S)) == cirq.CircuitDiagramInfo(wire_symbols=('@', 'S'), exponent=1)

    class UndiagrammableGate(cirq.testing.SingleQubitGate):

        def _has_unitary_(self):
            return True
    assert cirq.circuit_diagram_info(cirq.ControlledGate(UndiagrammableGate()), default=None) is None