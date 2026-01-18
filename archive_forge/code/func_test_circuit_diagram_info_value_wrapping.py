import numpy as np
import pytest
import sympy
import cirq
def test_circuit_diagram_info_value_wrapping():
    single_info = cirq.CircuitDiagramInfo(('Single',))

    class ReturnInfo:

        def _circuit_diagram_info_(self, args):
            return single_info

    class ReturnTuple:

        def _circuit_diagram_info_(self, args):
            return ('Single',)

    class ReturnList:

        def _circuit_diagram_info_(self, args):
            return ('Single' for _ in range(1))

    class ReturnGenerator:

        def _circuit_diagram_info_(self, args):
            return ['Single']

    class ReturnString:

        def _circuit_diagram_info_(self, args):
            return 'Single'
    assert cirq.circuit_diagram_info(ReturnInfo()) == cirq.circuit_diagram_info(ReturnTuple()) == cirq.circuit_diagram_info(ReturnString()) == cirq.circuit_diagram_info(ReturnList()) == cirq.circuit_diagram_info(ReturnGenerator()) == single_info
    double_info = cirq.CircuitDiagramInfo(('Single', 'Double'))

    class ReturnDoubleInfo:

        def _circuit_diagram_info_(self, args):
            return double_info

    class ReturnDoubleTuple:

        def _circuit_diagram_info_(self, args):
            return ('Single', 'Double')
    assert cirq.circuit_diagram_info(ReturnDoubleInfo()) == cirq.circuit_diagram_info(ReturnDoubleTuple()) == double_info