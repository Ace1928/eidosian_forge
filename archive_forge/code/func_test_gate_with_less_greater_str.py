import IPython.display
import numpy as np
import pytest
import cirq
from cirq.contrib.svg import circuit_to_svg
@pytest.mark.parametrize('symbol,svg_symbol', [('<a', '&lt;a'), ('<=b', '&lt;=b'), ('>c', '&gt;c'), ('>=d', '&gt;=d'), ('>e<', '&gt;e&lt;'), ('A[<virtual>]B[cirq.VirtualTag()]C>D<E', 'ABC&gt;D&lt;E')])
def test_gate_with_less_greater_str(symbol, svg_symbol):

    class CustomGate(cirq.Gate):

        def _num_qubits_(self) -> int:
            return 1

        def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
            return cirq.CircuitDiagramInfo(wire_symbols=[symbol])
    circuit = cirq.Circuit(CustomGate().on(cirq.LineQubit(0)))
    svg = circuit_to_svg(circuit)
    _ = IPython.display.SVG(svg)
    assert svg_symbol in svg