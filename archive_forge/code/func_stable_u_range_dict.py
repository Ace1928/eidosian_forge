from __future__ import annotations
import copy
import itertools
import random
import warnings
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from sympy import Symbol
from sympy.solvers import linsolve, solve
from pymatgen.analysis.wulff import WulffShape
from pymatgen.core import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.surface import get_slab_regions
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.vasp.outputs import Locpot, Outcar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.due import Doi, due
from pymatgen.util.plotting import pretty_plot
def stable_u_range_dict(self, chempot_range, ref_delu, no_doped=True, no_clean=False, delu_dict=None, miller_index=(), dmu_at_0=False, return_se_dict=False):
    """
        Creates a dictionary where each entry is a key pointing to a
        chemical potential range where the surface of that entry is stable.
        Does so by enumerating through all possible solutions (intersect)
        for surface energies of a specific facet.

        Args:
            chempot_range ([max_chempot, min_chempot]): Range to consider the
                stability of the slabs.
            ref_delu (sympy Symbol): The range stability of each slab is based
                on the chempot range of this chempot. Should be a sympy Symbol
                object of the format: Symbol("delu_el") where el is the name of
                the element
            no_doped (bool): Consider stability of clean slabs only.
            no_clean (bool): Consider stability of doped slabs only.
            delu_dict (dict): Dictionary of the chemical potentials to be set as
                constant. Note the key should be a sympy Symbol object of the
                format: Symbol("delu_el") where el is the name of the element.
            miller_index (list): Miller index for a specific facet to get a
                dictionary for.
            dmu_at_0 (bool): If True, if the surface energies corresponding to
                the chemical potential range is between a negative and positive
                value, the value is a list of three chemical potentials with the
                one in the center corresponding a surface energy of 0. Uselful
                in identifying unphysical ranges of surface energies and their
                chemical potential range.
            return_se_dict (bool): Whether or not to return the corresponding
                dictionary of surface energies
        """
    if delu_dict is None:
        delu_dict = {}
    chempot_range = sorted(chempot_range)
    stable_urange_dict, se_dict = ({}, {})
    for hkl in self.all_slab_entries:
        entries_in_hkl = []
        if miller_index and hkl != tuple(miller_index):
            continue
        if not no_clean:
            entries_in_hkl.extend(self.all_slab_entries[hkl])
        if not no_doped:
            for entry in self.all_slab_entries[hkl]:
                entries_in_hkl.extend(self.all_slab_entries[hkl][entry])
        for entry in entries_in_hkl:
            stable_urange_dict[entry] = []
            se_dict[entry] = []
        if len(entries_in_hkl) == 1:
            stable_urange_dict[entries_in_hkl[0]] = chempot_range
            u1, u2 = (delu_dict.copy(), delu_dict.copy())
            u1[ref_delu], u2[ref_delu] = (chempot_range[0], chempot_range[1])
            se = self.as_coeffs_dict[entries_in_hkl[0]]
            se_dict[entries_in_hkl[0]] = [sub_chempots(se, u1), sub_chempots(se, u2)]
            continue
        for pair in itertools.combinations(entries_in_hkl, 2):
            solution = self.get_surface_equilibrium(pair, delu_dict=delu_dict)
            if not solution:
                continue
            new_delu_dict = delu_dict.copy()
            new_delu_dict[ref_delu] = solution[ref_delu]
            stable_entry, gamma = self.get_stable_entry_at_u(hkl, new_delu_dict, no_doped=no_doped, no_clean=no_clean)
            if stable_entry not in pair:
                continue
            if not chempot_range[0] <= solution[ref_delu] <= chempot_range[1]:
                continue
            for entry in pair:
                stable_urange_dict[entry].append(solution[ref_delu])
                se_dict[entry].append(gamma)
        new_delu_dict = delu_dict.copy()
        for u in chempot_range:
            new_delu_dict[ref_delu] = u
            entry, gamma = self.get_stable_entry_at_u(hkl, delu_dict=new_delu_dict, no_doped=no_doped, no_clean=no_clean)
            stable_urange_dict[entry].append(u)
            se_dict[entry].append(gamma)
    if dmu_at_0:
        for entry, v in se_dict.items():
            if not stable_urange_dict[entry]:
                continue
            if v[0] * v[1] < 0:
                se = self.as_coeffs_dict[entry]
                v.append(0)
                stable_urange_dict[entry].append(solve(sub_chempots(se, delu_dict), ref_delu)[0])
    for entry, v in stable_urange_dict.items():
        se_dict[entry] = [se for idx, se in sorted(zip(v, se_dict[entry]))]
        stable_urange_dict[entry] = sorted(v)
    if return_se_dict:
        return (stable_urange_dict, se_dict)
    return stable_urange_dict