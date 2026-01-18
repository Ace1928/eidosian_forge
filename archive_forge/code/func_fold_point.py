from __future__ import annotations
import copy
import itertools
import logging
import math
import typing
import warnings
from collections import Counter
from typing import TYPE_CHECKING, Literal, cast, no_type_check
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import palettable
import scipy.interpolate as scint
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from monty.dev import requires
from monty.json import jsanitize
from pymatgen.core import Element
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
from pymatgen.electronic_structure.boltztrap import BoltztrapError
from pymatgen.electronic_structure.core import OrbitalType, Spin
from pymatgen.util.plotting import add_fig_kwargs, get_ax3d_fig, pretty_plot
def fold_point(p, lattice, coords_are_cartesian=False):
    """Folds a point with coordinates p inside the first Brillouin zone of the lattice.

    Args:
        p: coordinates of one point
        lattice: Lattice object used to convert from reciprocal to Cartesian coordinates
        coords_are_cartesian: Set to True if you are providing
            coordinates in Cartesian coordinates. Defaults to False.

    Returns:
        The Cartesian coordinates folded inside the first Brillouin zone
    """
    p = lattice.get_fractional_coords(p) if coords_are_cartesian else np.array(p)
    p = np.mod(p + 0.5 - 1e-10, 1) - 0.5 + 1e-10
    p = lattice.get_cartesian_coords(p)
    closest_lattice_point = None
    smallest_distance = 10000
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                lattice_point = np.dot((i, j, k), lattice.matrix)
                dist = np.linalg.norm(p - lattice_point)
                if closest_lattice_point is None or dist < smallest_distance:
                    closest_lattice_point = lattice_point
                    smallest_distance = dist
    if not np.allclose(closest_lattice_point, (0, 0, 0)):
        p = p - closest_lattice_point
    return p