from __future__ import annotations
import copy
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from warnings import warn
import numpy as np
from monty.json import MontyDecoder, MontyEncoder
from pymatgen.core import Molecule, Structure
from pymatgen.io.aims.inputs import AimsControlIn, AimsGeometryIn
from pymatgen.io.aims.parsers import AimsParseError, read_aims_output
from pymatgen.io.core import InputFile, InputGenerator, InputSet
def k2d(self, structure: Structure, k_grid: np.ndarray[int]):
    """Generate the kpoint density in each direction from given k_grid.

        Parameters
        ----------
        structure: Structure
            Contains unit cell and information about boundary conditions.
        k_grid: np.ndarray[int]
            k_grid that was used.

        Returns:
            dict: Density of kpoints in each direction. result.mean() computes average density
        """
    recipcell = structure.lattice.inv_matrix
    densities = k_grid / (2 * np.pi * np.sqrt((recipcell ** 2).sum(axis=1)))
    return np.array(densities)