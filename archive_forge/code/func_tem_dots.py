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
def tem_dots(self, structure: Structure, points) -> list:
    """
        Generates all TEM_dot as named tuples that will appear on the 2D diffraction pattern.

        Args:
            structure (Structure): The input structure.
            points (list): All points to be checked.

        Returns:
            list of TEM_dots
        """
    dots = []
    interplanar_spacings = self.get_interplanar_spacings(structure, points)
    bragg_angles = self.bragg_angles(interplanar_spacings)
    cell_intensity = self.normalized_cell_intensity(structure, bragg_angles)
    positions = self.get_positions(structure, points)
    for hkl, intensity in cell_intensity.items():
        dot = namedtuple('dot', ['position', 'hkl', 'intensity', 'film_radius', 'd_spacing'])
        position = positions[hkl]
        film_radius = 0.91 * (10 ** (-3) * self.cs * self.wavelength_rel() ** 3) ** Fraction('1/4')
        d_spacing = interplanar_spacings[hkl]
        tem_dot = dot(position, hkl, intensity, film_radius, d_spacing)
        dots.append(tem_dot)
    return dots