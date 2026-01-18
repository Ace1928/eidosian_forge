from __future__ import annotations
import copy
import warnings
from typing import TYPE_CHECKING
from monty.dev import requires
from pymatgen.core.structure import IMolecule, Molecule
def add_hydrogen(self) -> None:
    """Add hydrogens (make all hydrogen explicit)."""
    self._ob_mol.AddHydrogens()