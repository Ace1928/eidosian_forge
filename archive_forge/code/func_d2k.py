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
def d2k(self, structure: Structure, kptdensity: float | list[float]=5.0, even: bool=True) -> Iterable[float]:
    """Convert k-point density to Monkhorst-Pack grid size.

        inspired by [ase.calculators.calculator.kptdensity2monkhorstpack]

        Parameters
        ----------
        structure: Structure
            Contains unit cell and information about boundary conditions.
        kptdensity: float or list of floats
            Required k-point density.  Default value is 5.0 point per Ang^-1.
        even: bool
            Round up to even numbers.

        Returns:
            dict: Monkhorst-Pack grid size in all directions
        """
    recipcell = structure.lattice.inv_matrix
    return self.d2k_recipcell(recipcell, structure.lattice.pbc, kptdensity, even)