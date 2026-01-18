from __future__ import annotations
import copy
import warnings
from typing import TYPE_CHECKING
from monty.dev import requires
from pymatgen.core.structure import IMolecule, Molecule
def localopt(self, forcefield: str='mmff94', steps: int=500) -> None:
    """
        A wrapper to pybel's localopt method to optimize a Molecule.

        Args:
            forcefield: Default is mmff94. Options are 'gaff', 'ghemical',
                'mmff94', 'mmff94s', and 'uff'.
            steps: Default is 500.
        """
    pybelmol = pybel.Molecule(self._ob_mol)
    pybelmol.localopt(forcefield=forcefield, steps=steps)
    self._ob_mol = pybelmol.OBMol