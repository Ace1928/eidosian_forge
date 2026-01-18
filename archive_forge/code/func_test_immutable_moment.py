import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_immutable_moment():
    with pytest.raises(AttributeError):
        q1, q2 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(cirq.X(q1))
        moment = circuit.moments[0]
        moment.operations += (cirq.Y(q2),)