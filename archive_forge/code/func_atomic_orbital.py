import itertools
import collections
from pennylane import numpy as pnp
from .basis_data import atomic_numbers
from .basis_set import BasisFunction, mol_basis_data
from .integrals import contracted_norm, primitive_norm
def atomic_orbital(self, index):
    """Return a function that evaluates an atomic orbital at a given position.

        Args:
            index (int): index of the atomic orbital, order follwos the order of atomic symbols

        Returns:
            function: function that computes the value of the orbital at a given position

        **Example**

        >>> symbols  = ['H', 'H']
        >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
        >>> mol = qml.qchem.Molecule(symbols, geometry)
        >>> ao = mol.atomic_orbital(0)
        >>> ao(0.0, 0.0, 0.0)
        0.62824688
        """
    l = self.basis_set[index].l
    alpha = self.basis_set[index].alpha
    coeff = self.basis_set[index].coeff
    r = self.basis_set[index].r
    coeff = coeff * contracted_norm(l, alpha, coeff)
    lx, ly, lz = l

    def orbital(x, y, z):
        """Evaluate a basis function at a given position.

            Args:
                x (float): x component of the position
                y (float): y component of the position
                z (float): z component of the position

            Returns:
                array[float]: value of a basis function
            """
        c = (x - r[0]) ** lx * (y - r[1]) ** ly * (z - r[2]) ** lz
        e = [pnp.exp(-a * ((x - r[0]) ** 2 + (y - r[1]) ** 2 + (z - r[2]) ** 2)) for a in alpha]
        return c * pnp.dot(coeff, e)
    return orbital