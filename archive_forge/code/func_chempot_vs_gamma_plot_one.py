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
def chempot_vs_gamma_plot_one(self, ax: plt.Axes, entry: SlabEntry, ref_delu: Symbol, chempot_range: list[float], delu_dict: dict[Symbol, float] | None=None, delu_default: float=0, label: str='', JPERM2: bool=False) -> plt.Axes:
    """
        Helper function to help plot the surface energy of a
        single SlabEntry as a function of chemical potential.

        Args:
            ax (plt.Axes): Matplotlib Axes instance for plotting.
            entry: Entry of the slab whose surface energy we want
                to plot. (Add appropriate description for type)
            ref_delu (Symbol): The range stability of each slab is based
                on the chempot range of this chempot.
            chempot_range (list[float]): Range to consider the stability of the slabs.
            delu_dict (dict[Symbol, float]): Dictionary of the chemical potentials.
            delu_default (float): Default value for all unset chemical potentials.
            label (str): Label of the slab for the legend.
            JPERM2 (bool): Whether to plot surface energy in /m^2 (True) or
                eV/A^2 (False).

        Returns:
            plt.Axes: Plot of surface energy vs chemical potential for one entry.
        """
    delu_dict = delu_dict or {}
    chempot_range = sorted(chempot_range)
    ax = ax or plt.gca()
    ucell_comp = self.ucell_entry.composition.reduced_composition
    if entry.adsorbates:
        struct = entry.cleaned_up_slab
        clean_comp = struct.composition.reduced_composition
    else:
        clean_comp = entry.composition.reduced_composition
    mark = '--' if ucell_comp != clean_comp else '-'
    delu_dict = self.set_all_variables(delu_dict, delu_default)
    delu_dict[ref_delu] = chempot_range[0]
    gamma_min = self.as_coeffs_dict[entry]
    gamma_min = gamma_min if type(gamma_min).__name__ == 'float' else sub_chempots(gamma_min, delu_dict)
    delu_dict[ref_delu] = chempot_range[1]
    gamma_max = self.as_coeffs_dict[entry]
    gamma_max = gamma_max if type(gamma_max).__name__ == 'float' else sub_chempots(gamma_max, delu_dict)
    gamma_range = [gamma_min, gamma_max]
    se_range = np.array(gamma_range) * EV_PER_ANG2_TO_JOULES_PER_M2 if JPERM2 else gamma_range
    mark = entry.mark or mark
    color = entry.color or self.color_dict[entry]
    ax.plot(chempot_range, se_range, mark, color=color, label=label)
    return ax