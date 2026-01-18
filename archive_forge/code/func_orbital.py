import itertools
import collections
from pennylane import numpy as pnp
from .basis_data import atomic_numbers
from .basis_set import BasisFunction, mol_basis_data
from .integrals import contracted_norm, primitive_norm
def orbital(x, y, z):
    """Evaluate a molecular orbital at a given position.

            Args:
                x (float): x component of the position
                y (float): y component of the position
                z (float): z component of the position

            Returns:
                array[float]: value of a molecular orbital
            """
    m = 0.0
    for i in range(self.n_orbitals):
        m = m + c[i] * self.atomic_orbital(i)(x, y, z)
    return m