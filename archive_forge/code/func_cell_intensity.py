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
def cell_intensity(self, structure: Structure, bragg_angles: dict[tuple[int, int, int], float]) -> dict[tuple[int, int, int], float]:
    """
        Calculates cell intensity for each hkl plane. For simplicity's sake, take I = |F|**2.

        Args:
            structure (Structure): The input structure.
            bragg_angles (dict of 3-tuple to float): The Bragg angles for each hkl plane.

        Returns:
            dict of hkl plane to cell intensity
        """
    csf = self.cell_scattering_factors(structure, bragg_angles)
    csf_val = np.array(list(csf.values()))
    cell_intensity_val = (csf_val * csf_val.conjugate()).real
    return dict(zip(bragg_angles, cell_intensity_val))