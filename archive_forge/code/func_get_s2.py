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
def get_s2(self, bragg_angles: dict[tuple[int, int, int], float]) -> dict[tuple[int, int, int], float]:
    """
        Calculates the s squared parameter (= square of sin theta over lambda) for each hkl plane.

        Args:
            bragg_angles (dict): The bragg angles for each hkl plane.

        Returns:
            Dict of hkl plane to s2 parameter, calculates the s squared parameter
                (= square of sin theta over lambda).
        """
    plane = list(bragg_angles)
    bragg_angles_val = np.array(list(bragg_angles.values()))
    s2_val = (np.sin(bragg_angles_val) / self.wavelength_rel()) ** 2
    return dict(zip(plane, s2_val))