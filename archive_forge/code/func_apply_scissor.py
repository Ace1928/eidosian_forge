from __future__ import annotations
import collections
import itertools
import math
import re
import warnings
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.json import MSONable
from pymatgen.core import Element, Lattice, Structure, get_el_sp
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.coord import pbc_diff
def apply_scissor(self, new_band_gap):
    """Apply a scissor operator (shift of the CBM) to fit the given band gap.
        If it's a metal, we look for the band crossing the Fermi level
        and shift this one up. This will not work all the time for metals!

        Args:
            new_band_gap: the band gap the scissor band structure need to have.

        Returns:
            BandStructureSymmLine: with the applied scissor shift
        """
    if self.is_metal():
        max_index = -1000
        for idx in range(self.nb_bands):
            below = False
            above = False
            for j in range(len(self.kpoints)):
                if self.bands[Spin.up][idx][j] < self.efermi:
                    below = True
                if self.bands[Spin.up][idx][j] > self.efermi:
                    above = True
            if above and below and (idx > max_index):
                max_index = idx
            if self.is_spin_polarized:
                below = False
                above = False
                for j in range(len(self.kpoints)):
                    if self.bands[Spin.down][idx][j] < self.efermi:
                        below = True
                    if self.bands[Spin.down][idx][j] > self.efermi:
                        above = True
                if above and below and (idx > max_index):
                    max_index = idx
        old_dict = self.as_dict()
        shift = new_band_gap
        for spin in old_dict['bands']:
            for k in range(len(old_dict['bands'][spin])):
                for v in range(len(old_dict['bands'][spin][k])):
                    if k >= max_index:
                        old_dict['bands'][spin][k][v] = old_dict['bands'][spin][k][v] + shift
    else:
        shift = new_band_gap - self.get_band_gap()['energy']
        old_dict = self.as_dict()
        for spin in old_dict['bands']:
            for k in range(len(old_dict['bands'][spin])):
                for v in range(len(old_dict['bands'][spin][k])):
                    if old_dict['bands'][spin][k][v] >= old_dict['cbm']['energy']:
                        old_dict['bands'][spin][k][v] = old_dict['bands'][spin][k][v] + shift
        old_dict['efermi'] = old_dict['efermi'] + shift
    return self.from_dict(old_dict)