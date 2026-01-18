from __future__ import annotations
import os
import re
import subprocess
from monty.tempfile import ScratchDir
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.core import Element, Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def get_energy_relax_structure_buckingham(structure, gulp_cmd='gulp', keywords=('optimise', 'conp'), valence_dict=None):
    """Relax a structure and compute the energy using Buckingham potential.

    Args:
        structure: pymatgen Structure
        gulp_cmd: GULP command if not in standard place
        keywords: GULP first line keywords
        valence_dict: {El: valence}. Needed if the structure is not charge
            neutral.
    """
    gio = GulpIO()
    gc = GulpCaller(gulp_cmd)
    gin = gio.buckingham_input(structure, keywords, valence_dict=valence_dict)
    gout = gc.run(gin)
    energy = gio.get_energy(gout)
    relax_structure = gio.get_relaxed_structure(gout)
    return (energy, relax_structure)