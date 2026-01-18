from __future__ import annotations
import itertools
import logging
import warnings
from typing import TYPE_CHECKING
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
from pymatgen.core.structure import Structure
from pymatgen.util.coord import get_angle
from pymatgen.util.string import unicodeify_spacegroup
def hkl_tuple_to_str(hkl):
    """
    Prepare for display on plots "(hkl)" for surfaces

    Args:
        hkl: in the form of [h, k, l] or (h, k, l).
    """
    out = ''.join((f'\\overline{{{-x}}}' if x < 0 else str(x) for x in hkl))
    return f'(${out}$)'