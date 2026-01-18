import itertools
import collections
from pennylane import numpy as pnp
from .basis_data import atomic_numbers
from .basis_set import BasisFunction, mol_basis_data
from .integrals import contracted_norm, primitive_norm
def molecular_orbital(self, index):
    """Return a function that evaluates a molecular orbital at a given position.

        Args:
            index (int): index of the molecular orbital

        Returns:
            function: function to evaluate the molecular orbital

        **Example**

        >>> symbols  = ['H', 'H']
        >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
        >>> mol = qml.qchem.Molecule(symbols, geometry)
        >>> qml.qchem.scf(mol)() # run scf to obtain the optimized molecular orbitals
        >>> mo = mol.molecular_orbital(1)
        >>> mo(0.0, 0.0, 0.0)
        0.01825128
        """
    c = self.mo_coefficients[index]

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
    return orbital