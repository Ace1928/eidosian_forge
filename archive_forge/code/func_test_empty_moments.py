import IPython.display
import numpy as np
import pytest
import cirq
from cirq.contrib.svg import circuit_to_svg
def test_empty_moments():
    a, b = cirq.LineQubit.range(2)
    svg_1 = circuit_to_svg(cirq.Circuit(cirq.Moment(), cirq.Moment(cirq.CNOT(a, b)), cirq.Moment(), cirq.Moment(cirq.SWAP(a, b)), cirq.Moment(cirq.Z(a)), cirq.Moment(cirq.measure(a, b, key='z')), cirq.Moment()))
    assert '<svg' in svg_1
    assert '</svg>' in svg_1
    svg_2 = circuit_to_svg(cirq.Circuit(cirq.Moment()))
    assert '<svg' in svg_2
    assert '</svg>' in svg_2