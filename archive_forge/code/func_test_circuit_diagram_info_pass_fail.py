import numpy as np
import pytest
import sympy
import cirq
def test_circuit_diagram_info_pass_fail():

    class C:
        pass

    class D:

        def _circuit_diagram_info_(self, args):
            return NotImplemented

    class E:

        def _circuit_diagram_info_(self, args):
            return cirq.CircuitDiagramInfo(('X',))
    assert cirq.circuit_diagram_info(C(), default=None) is None
    assert cirq.circuit_diagram_info(D(), default=None) is None
    assert cirq.circuit_diagram_info(E(), default=None) == cirq.CircuitDiagramInfo(('X',))
    with pytest.raises(TypeError, match='no _circuit_diagram_info'):
        _ = cirq.circuit_diagram_info(C())
    with pytest.raises(TypeError, match='returned NotImplemented'):
        _ = cirq.circuit_diagram_info(D())
    assert cirq.circuit_diagram_info(E()) == cirq.CircuitDiagramInfo(('X',))