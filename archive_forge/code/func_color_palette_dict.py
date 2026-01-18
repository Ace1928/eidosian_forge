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
def color_palette_dict(self, alpha=0.35):
    """
        Helper function to assign each facet a unique color using a dictionary.

        Args:
            alpha (float): Degree of transparency

        return (dict): Dictionary of colors (r,g,b,a) when plotting surface
            energy stability. The keys are individual surface entries where
            clean surfaces have a solid color while the corresponding adsorbed
            surface will be transparent.
        """
    color_dict = {}
    for hkl in self.all_slab_entries:
        rgb_indices = [0, 1, 2]
        color = [0, 0, 0, 1]
        random.shuffle(rgb_indices)
        for idx, ind in enumerate(rgb_indices):
            if idx == 2:
                break
            color[ind] = np.random.uniform(0, 1)
        clean_list = np.linspace(0, 1, len(self.all_slab_entries[hkl]))
        for idx, clean in enumerate(self.all_slab_entries[hkl]):
            c = copy.copy(color)
            c[rgb_indices[2]] = clean_list[idx]
            color_dict[clean] = c
            for ads_entry in self.all_slab_entries[hkl][clean]:
                c_ads = copy.copy(c)
                c_ads[3] = alpha
                color_dict[ads_entry] = c_ads
    return color_dict