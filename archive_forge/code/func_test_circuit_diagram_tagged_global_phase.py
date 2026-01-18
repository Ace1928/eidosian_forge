from typing import AbstractSet, Iterator, Any
import pytest
import numpy as np
import sympy
import cirq
def test_circuit_diagram_tagged_global_phase():
    q = cirq.NamedQubit('a')
    global_phase = cirq.global_phase_operation(coefficient=-1.0).with_tags('tag0')
    assert cirq.circuit_diagram_info(global_phase, default='default') == 'default'
    cirq.testing.assert_has_diagram(cirq.Circuit(global_phase), "\n\nglobal phase:   π['tag0']", use_unicode_characters=True)
    cirq.testing.assert_has_diagram(cirq.Circuit(global_phase), '\n\nglobal phase:   π', use_unicode_characters=True, include_tags=False)
    expected = cirq.CircuitDiagramInfo(wire_symbols=(), exponent=1.0, connected=True, exponent_qubit_index=None, auto_exponent_parens=True)

    class NoWireSymbols(cirq.GlobalPhaseGate):

        def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> 'cirq.CircuitDiagramInfo':
            return expected
    no_wire_symbol_op = NoWireSymbols(coefficient=-1.0)().with_tags('tag0')
    assert cirq.circuit_diagram_info(no_wire_symbol_op, default='default') == expected
    cirq.testing.assert_has_diagram(cirq.Circuit(no_wire_symbol_op), "\n\nglobal phase:   π['tag0']", use_unicode_characters=True)
    tag1 = cirq.global_phase_operation(coefficient=1j).with_tags('tag1')
    tag2 = cirq.global_phase_operation(coefficient=1j).with_tags('tag2')
    c = cirq.Circuit([cirq.X(q), tag1, tag2])
    cirq.testing.assert_has_diagram(c, "a: ─────────────X───────────────────\n\nglobal phase:   π['tag1', 'tag2']", use_unicode_characters=True, precision=2)
    c = cirq.Circuit([cirq.X(q).with_tags('x_tag'), tag1])
    c.append(cirq.Moment([cirq.X(q), tag2]))
    cirq.testing.assert_has_diagram(c, "a: ─────────────X['x_tag']─────X──────────────\n\nglobal phase:   0.5π['tag1']   0.5π['tag2']\n", use_unicode_characters=True, include_tags=True)