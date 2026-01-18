from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import Matrix
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.qubit import IntQubit
from sympy.physics.quantum.grover import (apply_grover, superposition_basis,
def test_WGate():
    nqubits = 2
    basis_states = superposition_basis(nqubits)
    assert qapply(WGate(nqubits) * basis_states) == basis_states
    expected = 2 / sqrt(pow(2, nqubits)) * basis_states - IntQubit(1, nqubits=nqubits)
    assert qapply(WGate(nqubits) * IntQubit(1, nqubits=nqubits)) == expected