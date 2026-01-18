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
def get_first_point(self, structure: Structure, points: list) -> dict[tuple[int, int, int], float]:
    """
        Gets the first point to be plotted in the 2D DP, corresponding to maximum d/minimum R.

        Args:
            structure (Structure): The input structure.
            points (list): All points to be checked.

        Returns:
            dict of a hkl plane to max interplanar distance.
        """
    max_d = -100.0
    max_d_plane = (0, 0, 1)
    points = self.zone_axis_filter(points)
    spacings = self.get_interplanar_spacings(structure, points)
    for plane in sorted(spacings):
        if spacings[plane] > max_d:
            max_d_plane = plane
            max_d = spacings[plane]
    return {max_d_plane: max_d}