from __future__ import annotations
import os
import re
import subprocess
from monty.tempfile import ScratchDir
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.core import Element, Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def tersoff_input(self, structure: Structure, periodic=False, uc=True, *keywords):
    """Gets a GULP input with Tersoff potential for an oxide structure.

        Args:
            structure: pymatgen Structure
            periodic (Default=False): Flag denoting whether periodic
                boundary conditions are used
            library (Default=None): File containing the species and potential.
            uc (Default=True): Unit Cell Flag.
            keywords: GULP first line keywords.
        """
    gin = self.keyword_line(*keywords)
    gin += self.structure_lines(structure, cell_flg=periodic, frac_flg=periodic, anion_shell_flg=False, cation_shell_flg=False, symm_flg=not uc)
    gin += self.tersoff_potential(structure)
    return gin