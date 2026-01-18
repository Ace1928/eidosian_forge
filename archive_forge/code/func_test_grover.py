from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import Matrix
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.qubit import IntQubit
from sympy.physics.quantum.grover import (apply_grover, superposition_basis,
def test_grover():
    nqubits = 2
    assert apply_grover(return_one_on_one, nqubits) == IntQubit(1, nqubits=nqubits)
    nqubits = 4
    basis_states = superposition_basis(nqubits)
    expected = -13 * basis_states / 64 + 264 * IntQubit(2, nqubits) / 256
    assert apply_grover(return_one_on_two, 4) == qapply(expected)