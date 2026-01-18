import cirq
import cirq_ft
import pytest
from cirq_ft import infra
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_circuits():
    q = cirq.NamedQubit('q')
    circuit = cirq.Circuit(cirq.Rz(rads=0.6)(q), cirq.T(q), cirq.X(q) ** 0.5, cirq.Rx(rads=0.1)(q), cirq.Ry(rads=0.6)(q), cirq.measure(q, key='m'))
    assert cirq_ft.t_complexity(circuit) == cirq_ft.TComplexity(clifford=2, rotations=3, t=1)
    circuit = cirq.FrozenCircuit(cirq.T(q) ** (-1), cirq.Rx(rads=0.1)(q), cirq.measure(q, key='m'))
    assert cirq_ft.t_complexity(circuit) == cirq_ft.TComplexity(clifford=1, rotations=1, t=1)