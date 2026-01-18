from a list of entries within a chemical system containing 2 or more elements. The
from __future__ import annotations
import json
import os
import warnings
from functools import lru_cache
from itertools import groupby
from typing import TYPE_CHECKING
import numpy as np
import plotly.express as px
from monty.json import MSONable
from plotly.graph_objects import Figure, Mesh3d, Scatter, Scatter3d
from scipy.spatial import ConvexHull, HalfspaceIntersection
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core.composition import Composition, Element
from pymatgen.util.coord import Simplex
from pymatgen.util.due import Doi, due
from pymatgen.util.string import htmlify
def get_centroid_2d(vertices: np.ndarray) -> np.ndarray:
    """
    A bare-bones implementation of the formula for calculating the centroid of a 2D
    polygon. Useful for calculating the location of an annotation on a chemical
    potential domain within a 3D chemical potential diagram.

    NOTE vertices must be ordered circumferentially!

    Args:
        vertices: array of 2-d coordinates corresponding to a polygon, ordered
            circumferentially

    Returns:
        Array giving 2-d centroid coordinates
    """
    cx = 0
    cy = 0
    a = 0
    for idx in range(len(vertices) - 1):
        xi = vertices[idx, 0]
        yi = vertices[idx, 1]
        xi_p = vertices[idx + 1, 0]
        yi_p = vertices[idx + 1, 1]
        common_term = xi * yi_p - xi_p * yi
        cx += (xi + xi_p) * common_term
        cy += (yi + yi_p) * common_term
        a += common_term
    prefactor = 0.5 / (6 * a)
    return np.array([prefactor * cx, prefactor * cy])