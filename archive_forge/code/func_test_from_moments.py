import pytest
import sympy
import cirq
def test_from_moments():
    a, b, c, d = cirq.LineQubit.range(4)
    moment = cirq.Moment(cirq.Z(a), cirq.Z(b))
    subcircuit = cirq.FrozenCircuit.from_moments(cirq.X(c), cirq.Y(d))
    circuit = cirq.FrozenCircuit.from_moments(moment, subcircuit, [cirq.X(a), cirq.Y(b)], [cirq.X(c)], [], cirq.Z(d), [cirq.measure(a, b, key='ab'), cirq.measure(c, d, key='cd')])
    assert circuit == cirq.FrozenCircuit(cirq.Moment(cirq.Z(a), cirq.Z(b)), cirq.Moment(cirq.CircuitOperation(cirq.FrozenCircuit(cirq.Moment(cirq.X(c)), cirq.Moment(cirq.Y(d))))), cirq.Moment(cirq.X(a), cirq.Y(b)), cirq.Moment(cirq.X(c)), cirq.Moment(), cirq.Moment(cirq.Z(d)), cirq.Moment(cirq.measure(a, b, key='ab'), cirq.measure(c, d, key='cd')))
    assert circuit[0] is moment
    assert circuit[1].operations[0].circuit is subcircuit