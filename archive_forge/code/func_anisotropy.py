from __future__ import annotations
import itertools
import logging
import warnings
from typing import TYPE_CHECKING
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
from pymatgen.core.structure import Structure
from pymatgen.util.coord import get_angle
from pymatgen.util.string import unicodeify_spacegroup
@property
def anisotropy(self) -> float:
    """
        Returns:
            float: Coefficient of Variation from weighted surface energy. The ideal sphere is 0.
        """
    square_diff_energy = 0.0
    weighted_energy = self.weighted_surface_energy
    area_frac_dict = self.area_fraction_dict
    miller_energy_dict = self.miller_energy_dict
    for hkl, energy in miller_energy_dict.items():
        square_diff_energy += (energy - weighted_energy) ** 2 * area_frac_dict[hkl]
    return np.sqrt(square_diff_energy) / weighted_energy