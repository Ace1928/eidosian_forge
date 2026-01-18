from __future__ import annotations
from math import isclose
from typing import TYPE_CHECKING
import numpy as np
import scipy.constants as const
from pymatgen.core.units import kb as kb_ev
from pymatgen.util.due import Doi, due
def get_avg_mom_inertia(mol):
    """
    Calculate the average moment of inertia of a molecule

    Args:
        mol (Molecule): Pymatgen Molecule

    Returns:
        int, list: average moment of inertia, eigenvalues of the inertia tensor
    """
    centered_mol = mol.get_centered_molecule()
    inertia_tensor = np.zeros((3, 3))
    for site in centered_mol:
        c = site.coords
        wt = site.specie.atomic_mass
        for dim in range(3):
            inertia_tensor[dim, dim] += wt * (c[(dim + 1) % 3] ** 2 + c[(dim + 2) % 3] ** 2)
        for ii, jj in [(0, 1), (1, 2), (0, 2)]:
            inertia_tensor[ii, jj] += -wt * c[ii] * c[jj]
            inertia_tensor[jj, ii] += -wt * c[jj] * c[ii]
    inertia_eigen_vals = np.linalg.eig(inertia_tensor)[0] * amu_to_kg * 1e-20
    iav = np.average(inertia_eigen_vals)
    return (iav, inertia_eigen_vals)