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
def bragg_angles(self, interplanar_spacings: dict[tuple[int, int, int], float]) -> dict[tuple[int, int, int], float]:
    """
        Gets the Bragg angles for every hkl point passed in (where n = 1).

        Args:
            interplanar_spacings (dict): dictionary of hkl to interplanar spacing

        Returns:
            dict of hkl plane (3-tuple) to Bragg angle in radians (float)
        """
    plane = list(interplanar_spacings)
    interplanar_spacings_val = np.array(list(interplanar_spacings.values()))
    bragg_angles_val = np.arcsin(self.wavelength_rel() / (2 * interplanar_spacings_val))
    return dict(zip(plane, bragg_angles_val))