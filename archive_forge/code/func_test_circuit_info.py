import pickle
import numpy as np
import pytest
import cirq
def test_circuit_info():
    assert cirq.circuit_diagram_info(cirq.GridQubit(5, 2)) == cirq.CircuitDiagramInfo(wire_symbols=('(5, 2)',))
    assert cirq.circuit_diagram_info(cirq.GridQid(5, 2, dimension=3)) == cirq.CircuitDiagramInfo(wire_symbols=('(5, 2) (d=3)',))