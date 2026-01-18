from __future__ import annotations
import os
import warnings
from typing import TYPE_CHECKING, Any, Literal, cast
import numpy as np
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.local_env import LocalStructOrderParams, get_neighbors_of_site_with_index
from pymatgen.core import Species, get_el_sp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
@staticmethod
def get_magnitude_of_effect_from_spin_config(motif: str, spin_config: dict[str, float]) -> str:
    """
        Roughly, the magnitude of Jahn-Teller distortion will be:
        * in octahedral environments, strong if e_g orbitals
        unevenly occupied but weak if t_2g orbitals unevenly
        occupied
        * in tetrahedral environments always weaker.

        Args:
            motif: "oct" or "tet"
            spin_config: dict of 'e' (e_g) and 't' (t2_g) with number of electrons in each state

        Returns:
            str: "none", "weak" or "strong"
        """
    magnitude = 'none'
    if motif == 'oct':
        e_g = spin_config['e_g']
        t_2g = spin_config['t_2g']
        if e_g % 2 != 0 or t_2g % 3 != 0:
            magnitude = 'weak'
            if e_g % 2 == 1:
                magnitude = 'strong'
    elif motif == 'tet':
        e = spin_config['e']
        t_2 = spin_config['t_2']
        if e % 3 != 0 or t_2 % 2 != 0:
            magnitude = 'weak'
    return magnitude