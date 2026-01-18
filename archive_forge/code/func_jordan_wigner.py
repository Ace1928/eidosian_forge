from functools import singledispatch
from typing import Union
import pennylane as qml
from pennylane.operation import Operator
from pennylane.pauli import PauliSentence, PauliWord
from .fermionic import FermiSentence, FermiWord
def jordan_wigner(fermi_operator: Union[FermiWord, FermiSentence], ps: bool=False, wire_map: dict=None, tol: float=None) -> Union[Operator, PauliSentence]:
    """Convert a fermionic operator to a qubit operator using the Jordan-Wigner mapping.

    The fermionic creation and annihilation operators are mapped to the Pauli operators as

    .. math::

        a^{\\dagger}_0 =  \\left (\\frac{X_0 - iY_0}{2}  \\right ), \\:\\: \\text{...,} \\:\\:
        a^{\\dagger}_n = Z_0 \\otimes Z_1 \\otimes ... \\otimes Z_{n-1} \\otimes \\left (\\frac{X_n - iY_n}{2} \\right ),

    and

    .. math::

        a_0 =  \\left (\\frac{X_0 + iY_0}{2}  \\right ), \\:\\: \\text{...,} \\:\\:
        a_n = Z_0 \\otimes Z_1 \\otimes ... \\otimes Z_{n-1} \\otimes \\left (\\frac{X_n + iY_n}{2}  \\right ),

    where :math:`X`, :math:`Y`, and :math:`Z` are the Pauli operators.

    Args:
        fermi_operator(FermiWord, FermiSentence): the fermionic operator
        ps (bool): whether to return the result as a PauliSentence instead of an
            Operator. Defaults to False.
        wire_map (dict): a dictionary defining how to map the orbitals of
            the Fermi operator to qubit wires. If None, the integers used to
            order the orbitals will be used as wire labels. Defaults to None.
        tol (float): tolerance for discarding the imaginary part of the coefficients

    Returns:
        Union[PauliSentence, Operator]: a linear combination of qubit operators

    **Example**

    >>> w = FermiWord({(0, 0) : '+', (1, 1) : '-'})
    >>> jordan_wigner(w)
    (
        -0.25j * (Y(0) @ X(1))
      + (0.25+0j) * (Y(0) @ Y(1))
      + (0.25+0j) * (X(0) @ X(1))
      + 0.25j * (X(0) @ Y(1))
    )

    >>> jordan_wigner(w, ps=True)
    -0.25j * Y(0) @ X(1)
    + (0.25+0j) * Y(0) @ Y(1)
    + (0.25+0j) * X(0) @ X(1)
    + 0.25j * X(0) @ Y(1)

    >>> jordan_wigner(w, ps=True, wire_map={0: 2, 1: 3})
    -0.25j * Y(2) @ X(3)
    + (0.25+0j) * Y(2) @ Y(3)
    + (0.25+0j) * X(2) @ X(3)
    + 0.25j * X(2) @ Y(3)
    """
    return _jordan_wigner_dispatch(fermi_operator, ps, wire_map, tol)