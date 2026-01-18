from __future__ import annotations
import copy
import warnings
from typing import TYPE_CHECKING
from monty.dev import requires
from pymatgen.core.structure import IMolecule, Molecule
@property
def openbabel_mol(self):
    """Returns OpenBabel's OBMol."""
    return self._ob_mol