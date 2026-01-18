from __future__ import annotations
import os
import re
import subprocess
from monty.tempfile import ScratchDir
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.core import Element, Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
@staticmethod
def specie_potential_lines(structure, potential, **kwargs):
    """Generates GULP input specie and potential string for pymatgen
        structure.

        Args:
            structure: pymatgen Structure object
            potential: String specifying the type of potential used
            kwargs: Additional parameters related to potential. For
                potential == "buckingham",
                anion_shell_flg (default = False):
                If True, anions are considered polarizable.
                anion_core_chrg=float
                anion_shell_chrg=float
                cation_shell_flg (default = False):
                If True, cations are considered polarizable.
                cation_core_chrg=float
                cation_shell_chrg=float

        Returns:
            string containing specie and potential specification for gulp
            input.
        """
    raise NotImplementedError('gulp_specie_potential not yet implemented.\nUse library_line instead')