import pytest
import sympy
import cirq
import cirq_google as cg
from cirq_google.api import v2
def default_circuit():
    return cirq.FrozenCircuit(cirq.X(cirq.GridQubit(1, 1)) ** sympy.Symbol('k'), cirq.X(cirq.GridQubit(1, 2)).with_tags(DEFAULT_TOKEN), cirq.measure(cirq.GridQubit(1, 1), key='m'))