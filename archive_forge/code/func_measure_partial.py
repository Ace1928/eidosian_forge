import math
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.exponential import log
from sympy.core.basic import _sympify
from sympy.external.gmpy import SYMPY_INTS
from sympy.matrices import Matrix, zeros
from sympy.printing.pretty.stringpict import prettyForm
from sympy.physics.quantum.hilbert import ComplexSpace
from sympy.physics.quantum.state import Ket, Bra, State
from sympy.physics.quantum.qexpr import QuantumError
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.matrixutils import (
from mpmath.libmp.libintmath import bitcount
def measure_partial(qubit, bits, format='sympy', normalize=True):
    """Perform a partial ensemble measure on the specified qubits.

    Parameters
    ==========

    qubits : Qubit
        The qubit to measure.  This can be any Qubit or a linear combination
        of them.
    bits : tuple
        The qubits to measure.
    format : str
        The format of the intermediate matrices to use. Possible values are
        ('sympy','numpy','scipy.sparse'). Currently only 'sympy' is
        implemented.

    Returns
    =======

    result : list
        A list that consists of primitive states and their probabilities.

    Examples
    ========

        >>> from sympy.physics.quantum.qubit import Qubit, measure_partial
        >>> from sympy.physics.quantum.gate import H
        >>> from sympy.physics.quantum.qapply import qapply

        >>> c = H(0)*H(1)*Qubit('00')
        >>> c
        H(0)*H(1)*|00>
        >>> q = qapply(c)
        >>> measure_partial(q, (0,))
        [(sqrt(2)*|00>/2 + sqrt(2)*|10>/2, 1/2), (sqrt(2)*|01>/2 + sqrt(2)*|11>/2, 1/2)]
    """
    m = qubit_to_matrix(qubit, format)
    if isinstance(bits, (SYMPY_INTS, Integer)):
        bits = (int(bits),)
    if format == 'sympy':
        if normalize:
            m = m.normalized()
        possible_outcomes = _get_possible_outcomes(m, bits)
        output = []
        for outcome in possible_outcomes:
            prob_of_outcome = 0
            prob_of_outcome += (outcome.H * outcome)[0]
            if prob_of_outcome != 0:
                if normalize:
                    next_matrix = matrix_to_qubit(outcome.normalized())
                else:
                    next_matrix = matrix_to_qubit(outcome)
                output.append((next_matrix, prob_of_outcome))
        return output
    else:
        raise NotImplementedError('This function cannot handle non-SymPy matrix formats yet')