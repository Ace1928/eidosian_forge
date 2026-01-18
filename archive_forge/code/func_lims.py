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
@property
def lims(self) -> np.ndarray:
    """Returns array of limits used in constructing hyperplanes."""
    lims = np.array([[self.default_min_limit, 0]] * self.dim)
    for idx, elem in enumerate(self.elements):
        if self.limits and elem in self.limits:
            lims[idx, :] = self.limits[elem]
    return lims