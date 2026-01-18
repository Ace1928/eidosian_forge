from __future__ import annotations
import os
import re
import subprocess
from monty.tempfile import ScratchDir
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.core import Element, Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
@staticmethod
def structure_lines(structure: Structure, cell_flg: bool=True, frac_flg: bool=True, anion_shell_flg: bool=True, cation_shell_flg: bool=False, symm_flg: bool=True):
    """Generates GULP input string corresponding to pymatgen structure.

        Args:
            structure: pymatgen Structure object
            cell_flg (default = True): Option to use lattice parameters.
            frac_flg (default = True): If True, fractional coordinates
                are used. Else, Cartesian coordinates in Angstroms are used.
                ******
                GULP convention is to use fractional coordinates for periodic
                structures and Cartesian coordinates for non-periodic
                structures.
                ******
            anion_shell_flg (default = True): If True, anions are considered
                polarizable.
            cation_shell_flg (default = False): If True, cations are
                considered polarizable.
            symm_flg (default = True): If True, symmetry information is also
                written.

        Returns:
            string containing structure for GULP input
        """
    gin = ''
    if cell_flg:
        gin += 'cell\n'
        lattice = structure.lattice
        alpha, beta, gamma = lattice.angles
        a, b, c = lattice.lengths
        lat_str = f'{a:6f} {b:6f} {c:6f} {alpha:6f} {beta:6f} {gamma:6f}'
        gin += lat_str + '\n'
    if frac_flg:
        gin += 'frac\n'
        coords_key = 'frac_coords'
    else:
        gin += 'cart\n'
        coords_key = 'coords'
    for site in structure:
        coord = [str(i) for i in getattr(site, coords_key)]
        specie = site.specie
        core_site_desc = f'{specie.symbol} core {' '.join(coord)}\n'
        gin += core_site_desc
        if specie in _anions and anion_shell_flg or (specie in _cations and cation_shell_flg):
            shel_site_desc = f'{specie.symbol} shel {' '.join(coord)}\n'
            gin += shel_site_desc
        else:
            pass
    if symm_flg:
        gin += 'space\n'
        gin += str(SpacegroupAnalyzer(structure).get_space_group_number()) + '\n'
    return gin