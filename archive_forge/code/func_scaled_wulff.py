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
def scaled_wulff(self, wulff_shape, r):
    """
        Scales the Wulff shape with an effective radius r. Note that the resulting
            Wulff does not necessarily have the same effective radius as the one
            provided. The Wulff shape is scaled by its surface energies where first
            the surface energies are scale by the minimum surface energy and then
            multiplied by the given effective radius.

        Args:
            wulff_shape (WulffShape): Initial, unscaled WulffShape
            r (float): Arbitrary effective radius of the WulffShape

        Returns:
            WulffShape (scaled by r)
        """
    r_ratio = r / wulff_shape.effective_radius
    miller_list = list(wulff_shape.miller_energy_dict)
    se_list = np.array(list(wulff_shape.miller_energy_dict.values()))
    scaled_se = se_list * r_ratio
    return WulffShape(wulff_shape.lattice, miller_list, scaled_se, symprec=self.symprec)