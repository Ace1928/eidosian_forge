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
def area_fraction_dict(self) -> dict[tuple, float]:
    """
        Returns:
            dict: {hkl: area_hkl/total area on wulff}.
        """
    return {hkl: area / self.surface_area for hkl, area in self.miller_area_dict.items()}