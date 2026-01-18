from typing import Iterable, Union
import networkx as nx
import rustworkx as rx
import pennylane as qml
from pennylane import qaoa
def bit_driver(wires: Union[Iterable, qaoa.Wires], b: int):
    """Returns the bit-driver cost Hamiltonian.

    This Hamiltonian is defined as:

    .. math:: H \\ = \\ (-1)^{b + 1} \\displaystyle\\sum_{i} Z_i

    where :math:`Z_i` is the Pauli-Z operator acting on the
    :math:`i`-th wire and :math:`b \\ \\in \\ \\{0, \\ 1\\}`. This Hamiltonian is often used when
    constructing larger QAOA cost Hamiltonians.

    Args:
        wires (Iterable or Wires): The wires on which the Hamiltonian acts
        b (int): Either :math:`0` or :math:`1`. Determines whether the Hamiltonian assigns
                 lower energies to bitstrings with a majority of bits being :math:`0` or
                 a majority of bits being :math:`1`, respectively.

    Returns:
        .Hamiltonian:

    **Example**

    >>> wires = range(3)
    >>> hamiltonian = qaoa.bit_driver(wires, 1)
    >>> print(hamiltonian)
      (1) [Z0]
    + (1) [Z1]
    + (1) [Z2]
    """
    if b == 0:
        coeffs = [-1 for _ in wires]
    elif b == 1:
        coeffs = [1 for _ in wires]
    else:
        raise ValueError(f"'b' must be either 0 or 1, got {b}")
    ops = [qml.Z(w) for w in wires]
    return qml.Hamiltonian(coeffs, ops)