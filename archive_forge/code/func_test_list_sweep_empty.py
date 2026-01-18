import pytest
import sympy
import cirq
def test_list_sweep_empty():
    assert cirq.ListSweep([]).keys == []