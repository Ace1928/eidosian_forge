from typing import AbstractSet, Iterator, Any
import pytest
import numpy as np
import sympy
import cirq
def test_circuit_diagram():

    class TaggyTag:
        """Tag with a custom repr function to test circuit diagrams."""

        def __repr__(self):
            return 'TaggyTag()'
    h = cirq.H(cirq.GridQubit(1, 1))
    tagged_h = h.with_tags('tag1')
    non_string_tag_h = h.with_tags(TaggyTag())
    expected = cirq.CircuitDiagramInfo(wire_symbols=("H['tag1']",), exponent=1.0, connected=True, exponent_qubit_index=None, auto_exponent_parens=True)
    args = cirq.CircuitDiagramInfoArgs(None, None, None, None, None, False)
    assert cirq.circuit_diagram_info(tagged_h) == expected
    assert cirq.circuit_diagram_info(tagged_h, args) == cirq.circuit_diagram_info(h)
    c = cirq.Circuit(tagged_h)
    diagram_with_tags = "(1, 1): ───H['tag1']───"
    diagram_without_tags = '(1, 1): ───H───'
    assert str(cirq.Circuit(tagged_h)) == diagram_with_tags
    assert c.to_text_diagram() == diagram_with_tags
    assert c.to_text_diagram(include_tags=False) == diagram_without_tags
    c = cirq.Circuit(non_string_tag_h)
    diagram_with_non_string_tag = '(1, 1): ───H[TaggyTag()]───'
    assert c.to_text_diagram() == diagram_with_non_string_tag
    assert c.to_text_diagram(include_tags=False) == diagram_without_tags