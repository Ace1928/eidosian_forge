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
def matrix_to_qubit(matrix):
    """Convert from the matrix repr. to a sum of Qubit objects.

    Parameters
    ----------
    matrix : Matrix, numpy.matrix, scipy.sparse
        The matrix to build the Qubit representation of. This works with
        SymPy matrices, numpy matrices and scipy.sparse sparse matrices.

    Examples
    ========

    Represent a state and then go back to its qubit form:

        >>> from sympy.physics.quantum.qubit import matrix_to_qubit, Qubit
        >>> from sympy.physics.quantum.represent import represent
        >>> q = Qubit('01')
        >>> matrix_to_qubit(represent(q))
        |01>
    """
    format = 'sympy'
    if isinstance(matrix, numpy_ndarray):
        format = 'numpy'
    if isinstance(matrix, scipy_sparse_matrix):
        format = 'scipy.sparse'
    if matrix.shape[0] == 1:
        mlistlen = matrix.shape[1]
        nqubits = log(mlistlen, 2)
        ket = False
        cls = QubitBra
    elif matrix.shape[1] == 1:
        mlistlen = matrix.shape[0]
        nqubits = log(mlistlen, 2)
        ket = True
        cls = Qubit
    else:
        raise QuantumError('Matrix must be a row/column vector, got %r' % matrix)
    if not isinstance(nqubits, Integer):
        raise QuantumError('Matrix must be a row/column vector of size 2**nqubits, got: %r' % matrix)
    result = 0
    for i in range(mlistlen):
        if ket:
            element = matrix[i, 0]
        else:
            element = matrix[0, i]
        if format in ('numpy', 'scipy.sparse'):
            element = complex(element)
        if element != 0.0:
            qubit_array = [int(i & 1 << x != 0) for x in range(nqubits)]
            qubit_array.reverse()
            result = result + element * cls(*qubit_array)
    if isinstance(result, (Mul, Add, Pow)):
        result = result.expand()
    return result