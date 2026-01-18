import os
import copy
import subprocess
from math import pi, sqrt
import pathlib
from typing import Union, Optional, List, Set, Dict, Any
import warnings
import numpy as np
from ase.cell import Cell
from ase.outputs import Properties, all_outputs
from ase.utils import jsonable
from ase.calculators.abc import GetPropertiesMixin
def calculate_numerical_stress(self, atoms, d=1e-06, voigt=True):
    """Calculate numerical stress using finite difference."""
    stress = np.zeros((3, 3), dtype=float)
    cell = atoms.cell.copy()
    V = atoms.get_volume()
    for i in range(3):
        x = np.eye(3)
        x[i, i] += d
        atoms.set_cell(np.dot(cell, x), scale_atoms=True)
        eplus = atoms.get_potential_energy(force_consistent=True)
        x[i, i] -= 2 * d
        atoms.set_cell(np.dot(cell, x), scale_atoms=True)
        eminus = atoms.get_potential_energy(force_consistent=True)
        stress[i, i] = (eplus - eminus) / (2 * d * V)
        x[i, i] += d
        j = i - 2
        x[i, j] = d
        x[j, i] = d
        atoms.set_cell(np.dot(cell, x), scale_atoms=True)
        eplus = atoms.get_potential_energy(force_consistent=True)
        x[i, j] = -d
        x[j, i] = -d
        atoms.set_cell(np.dot(cell, x), scale_atoms=True)
        eminus = atoms.get_potential_energy(force_consistent=True)
        stress[i, j] = (eplus - eminus) / (4 * d * V)
        stress[j, i] = stress[i, j]
    atoms.set_cell(cell, scale_atoms=True)
    if voigt:
        return stress.flat[[0, 4, 8, 5, 2, 1]]
    else:
        return stress