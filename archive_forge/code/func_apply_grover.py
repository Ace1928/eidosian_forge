from sympy.core.numbers import pi
from sympy.core.sympify import sympify
from sympy.core.basic import Atom
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import eye
from sympy.core.numbers import NegativeOne
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.qexpr import QuantumError
from sympy.physics.quantum.hilbert import ComplexSpace
from sympy.physics.quantum.operator import UnitaryOperator
from sympy.physics.quantum.gate import Gate
from sympy.physics.quantum.qubit import IntQubit
def apply_grover(oracle, nqubits, iterations=None):
    """Applies grover's algorithm.

    Parameters
    ==========

    oracle : callable
        The unknown callable function that returns true when applied to the
        desired qubits and false otherwise.

    Returns
    =======

    state : Expr
        The resulting state after Grover's algorithm has been iterated.

    Examples
    ========

    Apply grover's algorithm to an even superposition of 2 qubits::

        >>> from sympy.physics.quantum.qapply import qapply
        >>> from sympy.physics.quantum.qubit import IntQubit
        >>> from sympy.physics.quantum.grover import apply_grover
        >>> f = lambda qubits: qubits == IntQubit(2)
        >>> qapply(apply_grover(f, 2))
        |2>

    """
    if nqubits <= 0:
        raise QuantumError("Grover's algorithm needs nqubits > 0, received %r qubits" % nqubits)
    if iterations is None:
        iterations = floor(sqrt(2 ** nqubits) * (pi / 4))
    v = OracleGate(nqubits, oracle)
    iterated = superposition_basis(nqubits)
    for iter in range(iterations):
        iterated = grover_iteration(iterated, v)
        iterated = qapply(iterated)
    return iterated