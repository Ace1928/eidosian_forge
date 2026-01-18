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
def get_2d_orthonormal_vector(line_pts: np.ndarray) -> np.ndarray:
    """
    Calculates a vector that is orthonormal to a line given by a set of points. Used
    for determining the location of an annotation on a 2-d chemical potential diagram.

    Args:
        line_pts: a 2x2 array in the form of [[x0, y0], [x1, y1]] giving the
            coordinates of a line

    Returns:
        np.ndarray: A length-2 vector that is orthonormal to the line.
    """
    x = line_pts[:, 0]
    y = line_pts[:, 1]
    x_diff = abs(x[1] - x[0])
    y_diff = abs(y[1] - y[0])
    theta = np.pi / 2 if np.isclose(x_diff, 0) else np.arctan(y_diff / x_diff)
    return np.array([np.sin(theta), np.cos(theta)])