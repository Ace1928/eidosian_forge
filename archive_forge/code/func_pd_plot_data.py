from __future__ import annotations
import collections
import itertools
import json
import logging
import math
import os
import re
import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal, no_type_check
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib import cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.font_manager import FontProperties
from monty.json import MontyDecoder, MSONable
from scipy import interpolate
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from tqdm import tqdm
from pymatgen.analysis.reaction_calculator import Reaction, ReactionError
from pymatgen.core import DummySpecies, Element, get_el_sp
from pymatgen.core.composition import Composition
from pymatgen.entries import Entry
from pymatgen.util.coord import Simplex, in_coord_list
from pymatgen.util.due import Doi, due
from pymatgen.util.plotting import pretty_plot
from pymatgen.util.string import htmlify, latexify
@property
@lru_cache(1)
def pd_plot_data(self):
    """
        Plotting data for phase diagram. Cached for repetitive calls.

        2-comp - Full hull with energies
        3/4-comp - Projection into 2D or 3D Gibbs triangles

        Returns:
            A tuple containing three objects (lines, stable_entries, unstable_entries):
            - lines is a list of list of coordinates for lines in the PD.
            - stable_entries is a dict of {coordinates : entry} for each stable node
                in the phase diagram. (Each coordinate can only have one
                stable phase)
            - unstable_entries is a dict of {entry: coordinates} for all unstable
                nodes in the phase diagram.
        """
    pd = self._pd
    entries = pd.qhull_entries
    data = np.array(pd.qhull_data)
    lines = []
    stable_entries = {}
    for line in self.lines:
        entry1 = entries[line[0]]
        entry2 = entries[line[1]]
        if self._dim < 3:
            x = [data[line[0]][0], data[line[1]][0]]
            y = [pd.get_form_energy_per_atom(entry1), pd.get_form_energy_per_atom(entry2)]
            coord = [x, y]
        elif self._dim == 3:
            coord = triangular_coord(data[line, 0:2])
        else:
            coord = tet_coord(data[line, 0:3])
        lines.append(coord)
        labelcoord = list(zip(*coord))
        stable_entries[labelcoord[0]] = entry1
        stable_entries[labelcoord[1]] = entry2
    all_entries = pd.all_entries
    all_data = np.array(pd.all_entries_hulldata)
    unstable_entries = {}
    stable = pd.stable_entries
    for idx, entry in enumerate(all_entries):
        if entry not in stable:
            if self._dim < 3:
                x = [all_data[idx][0], all_data[idx][0]]
                y = [pd.get_form_energy_per_atom(entry), pd.get_form_energy_per_atom(entry)]
                coord = [x, y]
            elif self._dim == 3:
                coord = triangular_coord([all_data[idx, 0:2], all_data[idx, 0:2]])
            else:
                coord = tet_coord([all_data[idx, 0:3], all_data[idx, 0:3], all_data[idx, 0:3]])
            labelcoord = list(zip(*coord))
            unstable_entries[entry] = labelcoord[0]
    return (lines, stable_entries, unstable_entries)