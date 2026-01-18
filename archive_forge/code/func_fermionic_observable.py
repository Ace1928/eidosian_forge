import numpy as np
import pennylane as qml
from pennylane.fermi import FermiSentence, FermiWord
from pennylane.operation import active_new_opmath
from pennylane.pauli.utils import simplify
def fermionic_observable(constant, one=None, two=None, cutoff=1e-12):
    """Create a fermionic observable from molecular orbital integrals.

    Args:
        constant (array[float]): the contribution of the core orbitals and nuclei
        one (array[float]): the one-particle molecular orbital integrals
        two (array[float]): the two-particle molecular orbital integrals
        cutoff (float): cutoff value for discarding the negligible integrals

    Returns:
        FermiSentence: fermionic observable

    **Example**

    >>> constant = np.array([1.0])
    >>> integral = np.array([[0.5, -0.8270995], [-0.8270995, 0.5]])
    >>> fermionic_observable(constant, integral)
    1.0 * I
    + 0.5 * a⁺(0) a(0)
    + -0.8270995 * a⁺(0) a(2)
    + 0.5 * a⁺(1) a(1)
    + -0.8270995 * a⁺(1) a(3)
    + -0.8270995 * a⁺(2) a(0)
    + 0.5 * a⁺(2) a(2)
    + -0.8270995 * a⁺(3) a(1)
    + 0.5 * a⁺(3) a(3)
    """
    coeffs = qml.math.array([])
    if not qml.math.allclose(constant, 0.0):
        coeffs = qml.math.concatenate((coeffs, constant))
        operators = [[]]
    else:
        operators = []
    if one is not None:
        indices_one = qml.math.argwhere(abs(one) >= cutoff)
        operators_one = (indices_one * 2).tolist() + (indices_one * 2 + 1).tolist()
        coeffs_one = qml.math.tile(one[abs(one) >= cutoff], 2)
        coeffs = qml.math.convert_like(coeffs, one)
        coeffs = qml.math.concatenate((coeffs, coeffs_one))
        operators = operators + operators_one
    if two is not None:
        indices_two = np.array(qml.math.argwhere(abs(two) >= cutoff))
        n = len(indices_two)
        operators_two = [(indices_two[i] * 2).tolist() for i in range(n)] + [(indices_two[i] * 2 + [0, 1, 1, 0]).tolist() for i in range(n)] + [(indices_two[i] * 2 + [1, 0, 0, 1]).tolist() for i in range(n)] + [(indices_two[i] * 2 + 1).tolist() for i in range(n)]
        coeffs_two = qml.math.tile(two[abs(two) >= cutoff], 4) / 2
        coeffs = qml.math.concatenate((coeffs, coeffs_two))
        operators = operators + operators_two
    indices_sort = [operators.index(i) for i in sorted(operators)]
    if indices_sort:
        indices_sort = qml.math.array(indices_sort)
    sentence = FermiSentence({FermiWord({}): constant[0]})
    for c, o in zip(coeffs[indices_sort], sorted(operators)):
        if len(o) == 2:
            sentence.update({FermiWord({(0, o[0]): '+', (1, o[1]): '-'}): c})
        if len(o) == 4:
            sentence.update({FermiWord({(0, o[0]): '+', (1, o[1]): '+', (2, o[2]): '-', (3, o[3]): '-'}): c})
    sentence.simplify()
    return sentence