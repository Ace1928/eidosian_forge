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
def get_stable_entry_at_u(self, miller_index, delu_dict=None, delu_default=0, no_doped=False, no_clean=False):
    """
        Returns the entry corresponding to the most stable slab for a particular
            facet at a specific chempot. We assume that surface energy is constant
            so all free variables must be set with delu_dict, otherwise they are
            assumed to be equal to delu_default.

        Args:
            miller_index ((h,k,l)): The facet to find the most stable slab in
            delu_dict (dict): Dictionary of the chemical potentials to be set as
                constant. Note the key should be a sympy Symbol object of the
                format: Symbol("delu_el") where el is the name of the element.
            delu_default (float): Default value for all unset chemical potentials
            no_doped (bool): Consider stability of clean slabs only.
            no_clean (bool): Consider stability of doped slabs only.

        Returns:
            SlabEntry, surface_energy (float)
        """
    all_delu_dict = self.set_all_variables(delu_dict, delu_default)

    def get_coeffs(e):
        coeffs = []
        for du in all_delu_dict:
            if type(self.as_coeffs_dict[e]).__name__ == 'float':
                coeffs.append(self.as_coeffs_dict[e])
            elif du in self.as_coeffs_dict[e]:
                coeffs.append(self.as_coeffs_dict[e][du])
            else:
                coeffs.append(0)
        return np.array(coeffs)
    all_entries, all_coeffs = ([], [])
    for entry in self.all_slab_entries[miller_index]:
        if not no_clean:
            all_entries.append(entry)
            all_coeffs.append(get_coeffs(entry))
        if not no_doped:
            for ads_entry in self.all_slab_entries[miller_index][entry]:
                all_entries.append(ads_entry)
                all_coeffs.append(get_coeffs(ads_entry))
    du_vals = np.array(list(all_delu_dict.values()))
    all_gamma = list(np.dot(all_coeffs, du_vals.T))
    return (all_entries[all_gamma.index(min(all_gamma))], float(min(all_gamma)))