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
def shape_factor(self) -> float:
    """
        This is useful for determining the critical nucleus size.
        A large shape factor indicates great anisotropy.
        See Ballufi, R. W., Allen, S. M. & Carter, W. C. Kinetics
            of Materials. (John Wiley & Sons, 2005), p.461.

        Returns:
            float: Shape factor.
        """
    return self.surface_area / self.volume ** (2 / 3)