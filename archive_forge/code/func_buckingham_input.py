from __future__ import annotations
import os
import re
import subprocess
from monty.tempfile import ScratchDir
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.core import Element, Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def buckingham_input(self, structure: Structure, keywords, library=None, uc=True, valence_dict=None):
    """Gets a GULP input for an oxide structure and buckingham potential
        from library.

        Args:
            structure: pymatgen Structure
            keywords: GULP first line keywords.
            library (Default=None): File containing the species and potential.
            uc (Default=True): Unit Cell Flag.
            valence_dict: {El: valence}
        """
    gin = self.keyword_line(*keywords)
    gin += self.structure_lines(structure, symm_flg=not uc)
    if not library:
        gin += self.buckingham_potential(structure, valence_dict)
    else:
        gin += self.library_line(library)
    return gin