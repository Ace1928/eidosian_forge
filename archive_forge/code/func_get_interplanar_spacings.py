from __future__ import annotations
import json
import os
from collections import namedtuple
from fractions import Fraction
from typing import TYPE_CHECKING, cast
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.constants as sc
from pymatgen.analysis.diffraction.core import AbstractDiffractionPatternCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.string import latexify_spacegroup, unicodeify_spacegroup
def get_interplanar_spacings(self, structure: Structure, points: list[tuple[int, int, int]] | np.ndarray) -> dict[tuple[int, int, int], float]:
    """
        Args:
            structure (Structure): the input structure.
            points (tuple): the desired hkl indices.

        Returns:
            Dict of hkl to its interplanar spacing, in angstroms (float).
        """
    points_filtered = self.zone_axis_filter(points)
    if (0, 0, 0) in points_filtered:
        points_filtered.remove((0, 0, 0))
    interplanar_spacings_val = np.array([structure.lattice.d_hkl(x) for x in points_filtered])
    return dict(zip(points_filtered, interplanar_spacings_val))