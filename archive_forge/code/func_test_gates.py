import cirq
import cirq_ft
import pytest
from cirq_ft import infra
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_gates():
    assert cirq_ft.t_complexity(cirq.T) == cirq_ft.TComplexity(t=1)
    assert cirq_ft.t_complexity(cirq.T ** (-1)) == cirq_ft.TComplexity(t=1)
    assert cirq_ft.t_complexity(cirq.H) == cirq_ft.TComplexity(clifford=1)
    assert cirq_ft.t_complexity(cirq.CNOT) == cirq_ft.TComplexity(clifford=1)
    assert cirq_ft.t_complexity(cirq.S) == cirq_ft.TComplexity(clifford=1)
    assert cirq_ft.t_complexity(cirq.S ** (-1)) == cirq_ft.TComplexity(clifford=1)
    assert cirq_ft.t_complexity(cirq.X) == cirq_ft.TComplexity(clifford=1)
    assert cirq_ft.t_complexity(cirq.Y) == cirq_ft.TComplexity(clifford=1)
    assert cirq_ft.t_complexity(cirq.Z) == cirq_ft.TComplexity(clifford=1)
    assert cirq_ft.t_complexity(cirq.Rx(rads=2)) == cirq_ft.TComplexity(rotations=1)
    assert cirq_ft.t_complexity(cirq.Ry(rads=2)) == cirq_ft.TComplexity(rotations=1)
    assert cirq_ft.t_complexity(cirq.Rz(rads=2)) == cirq_ft.TComplexity(rotations=1)
    assert cirq_ft.t_complexity(cirq_ft.And()) == cirq_ft.TComplexity(t=4, clifford=9)
    assert cirq_ft.t_complexity(cirq_ft.And() ** (-1)) == cirq_ft.TComplexity(clifford=4)
    assert cirq_ft.t_complexity(cirq.FREDKIN) == cirq_ft.TComplexity(t=7, clifford=10)