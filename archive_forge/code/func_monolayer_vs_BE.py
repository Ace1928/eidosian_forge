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
def monolayer_vs_BE(self, plot_eads=False):
    """
        Plots the binding energy as a function of monolayers (ML), i.e.
            the fractional area adsorbate density for all facets. For each
            facet at a specific monolayer, only plot the lowest binding energy.

        Args:
            plot_eads (bool): Option to plot the adsorption energy (binding
                energy multiplied by number of adsorbates) instead.

        Returns:
            Plot: Plot of binding energy vs monolayer for all facets.
        """
    ax = pretty_plot(width=8, height=7)
    for hkl in self.all_slab_entries:
        ml_be_dict = {}
        for clean_entry in self.all_slab_entries[hkl]:
            if self.all_slab_entries[hkl][clean_entry]:
                for ads_entry in self.all_slab_entries[hkl][clean_entry]:
                    if ads_entry.get_monolayer not in ml_be_dict:
                        ml_be_dict[ads_entry.get_monolayer] = 1000
                    be = ads_entry.gibbs_binding_energy(eads=plot_eads)
                    if be < ml_be_dict[ads_entry.get_monolayer]:
                        ml_be_dict[ads_entry.get_monolayer] = be
        vals = sorted(ml_be_dict.items())
        monolayers, BEs = zip(*vals)
        ax.plot(monolayers, BEs, '-o', c=self.color_dict[clean_entry], label=hkl)
    adsorbates = tuple(ads_entry.ads_entries_dict)
    ax.set_xlabel(f'{' '.join(adsorbates)} Coverage (ML)')
    ax.set_ylabel('Adsorption Energy (eV)' if plot_eads else 'Binding Energy (eV)')
    ax.legend()
    plt.tight_layout()
    return ax