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
def get_locpot_along_slab_plot(self, label_energies=True, plt=None, label_fontsize=10):
    """
        Returns a plot of the local potential (eV) vs the
            position along the c axis of the slab model (Ang).

        Args:
            label_energies (bool): Whether to label relevant energy
                quantities such as the work function, Fermi energy,
                vacuum locpot, bulk-like locpot
            plt (plt): Matplotlib pyplot object
            label_fontsize (float): Fontsize of labels

        Returns plt of the locpot vs c axis
        """
    plt = plt or pretty_plot(width=6, height=4)
    plt.plot(self.along_c, self.locpot_along_c, 'b--')
    xg, yg = ([], [])
    for idx, pot in enumerate(self.locpot_along_c):
        in_slab = False
        for r in self.slab_regions:
            if r[0] <= self.along_c[idx] <= r[1]:
                in_slab = True
        if len(self.slab_regions) > 1:
            if self.along_c[idx] >= self.slab_regions[1][1]:
                in_slab = True
            if self.along_c[idx] <= self.slab_regions[0][0]:
                in_slab = True
        if in_slab or pot < self.ave_bulk_p:
            yg.append(self.ave_bulk_p)
            xg.append(self.along_c[idx])
        else:
            yg.append(pot)
            xg.append(self.along_c[idx])
    xg, yg = zip(*sorted(zip(xg, yg)))
    plt.plot(xg, yg, 'r', linewidth=2.5, zorder=-1)
    if label_energies:
        plt = self.get_labels(plt, label_fontsize=label_fontsize)
    plt.xlim([0, 1])
    plt.ylim([min(self.locpot_along_c), self.vacuum_locpot + self.ave_locpot * 0.2])
    plt.xlabel('Fractional coordinates ($\\hat{c}$)', fontsize=25)
    plt.xticks(fontsize=15, rotation=45)
    plt.ylabel('Potential (eV)', fontsize=25)
    plt.yticks(fontsize=15)
    return plt