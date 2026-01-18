import random
from sympy.core.numbers import (Integer, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import Matrix
from sympy.physics.quantum.qubit import (measure_all, measure_partial,
from sympy.physics.quantum.gate import (HadamardGate, CNOT, XGate, YGate,
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.shor import Qubit
from sympy.testing.pytest import raises
from sympy.physics.quantum.density import Density
from sympy.physics.quantum.trace import Tr
def test_apply_represent_equality():
    gates = [HadamardGate(int(3 * random.random())), XGate(int(3 * random.random())), ZGate(int(3 * random.random())), YGate(int(3 * random.random())), ZGate(int(3 * random.random())), PhaseGate(int(3 * random.random()))]
    circuit = Qubit(int(random.random() * 2), int(random.random() * 2), int(random.random() * 2), int(random.random() * 2), int(random.random() * 2), int(random.random() * 2))
    for i in range(int(random.random() * 6)):
        circuit = gates[int(random.random() * 6)] * circuit
    mat = represent(circuit, nqubits=6)
    states = qapply(circuit)
    state_rep = matrix_to_qubit(mat)
    states = states.expand()
    state_rep = state_rep.expand()
    assert state_rep == states