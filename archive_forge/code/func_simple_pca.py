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
def simple_pca(data: np.ndarray, k: int=2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    A bare-bones implementation of principal component analysis (PCA) used in the
    ChemicalPotentialDiagram class for plotting.

    Args:
        data: array of observations
        k: Number of principal components returned

    Returns:
        tuple: projected data, eigenvalues, eigenvectors
    """
    data = data - np.mean(data.T, axis=1)
    cov = np.cov(data.T)
    v, w = np.linalg.eig(cov)
    idx = v.argsort()[::-1]
    v = v[idx]
    w = w[:, idx]
    scores = data.dot(w[:, :k])
    return (scores, v[:k], w[:, :k])