import cirq
import cirq_ft
import cirq_ft.infra.testing as cq_testing
import IPython.display
import ipywidgets
import pytest
from cirq_ft.infra.jupyter_tools import display_gate_and_compilation, svg_circuit
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_display_gate_and_compilation(monkeypatch):
    call_args = []

    def _mock_display(stuff):
        call_args.append(stuff)
    monkeypatch.setattr(IPython.display, 'display', _mock_display)
    g = cq_testing.GateHelper(cirq_ft.And(cv=(1, 1, 1)))
    display_gate_and_compilation(g)
    display_arg, = call_args
    assert isinstance(display_arg, ipywidgets.HBox)
    assert len(display_arg.children) == 2