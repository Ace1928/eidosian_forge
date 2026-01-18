import cirq
import cirq_ft
import pytest
from cirq_ft import infra
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_classically_controlled_operations():
    q = cirq.NamedQubit('q')
    assert cirq_ft.t_complexity(cirq.X(q).with_classical_controls('c')) == cirq_ft.TComplexity(clifford=1)
    assert cirq_ft.t_complexity(cirq.Rx(rads=0.1)(q).with_classical_controls('c')) == cirq_ft.TComplexity(rotations=1)
    assert cirq_ft.t_complexity(cirq.T(q).with_classical_controls('c')) == cirq_ft.TComplexity(t=1)