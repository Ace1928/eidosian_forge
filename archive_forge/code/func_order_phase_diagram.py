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
def order_phase_diagram(lines, stable_entries, unstable_entries, ordering):
    """
    Orders the entries (their coordinates) in a phase diagram plot according
    to the user specified ordering.
    Ordering should be given as ['Up', 'Left', 'Right'], where Up,
    Left and Right are the names of the entries in the upper, left and right
    corners of the triangle respectively.

    Args:
        lines: list of list of coordinates for lines in the PD.
        stable_entries: {coordinate : entry} for each stable node in the
            phase diagram. (Each coordinate can only have one stable phase)
        unstable_entries: {entry: coordinates} for all unstable nodes in the
            phase diagram.
        ordering: Ordering of the phase diagram, given as a list ['Up',
            'Left','Right']

    Returns:
        tuple[list, dict, dict]:
            - new_lines is a list of list of coordinates for lines in the PD.
            - new_stable_entries is a {coordinate: entry} for each stable node
            in the phase diagram. (Each coordinate can only have one
            stable phase)
            - new_unstable_entries is a {entry: coordinates} for all unstable
            nodes in the phase diagram.
    """
    yup = -1000.0
    xleft = 1000.0
    xright = -1000.0
    for coord in stable_entries:
        if coord[0] > xright:
            xright = coord[0]
            nameright = stable_entries[coord].name
        if coord[0] < xleft:
            xleft = coord[0]
            nameleft = stable_entries[coord].name
        if coord[1] > yup:
            yup = coord[1]
            nameup = stable_entries[coord].name
    if nameup not in ordering or nameright not in ordering or nameleft not in ordering:
        raise ValueError(f'Error in ordering_phase_diagram :\n{nameup!r}, {nameleft!r} and {nameright!r} should be in ordering : {ordering}')
    cc = np.array([0.5, np.sqrt(3.0) / 6.0], float)
    if nameup == ordering[0]:
        if nameleft == ordering[1]:
            return (lines, stable_entries, unstable_entries)
        new_lines = [[np.array(1 - x), y] for x, y in lines]
        new_stable_entries = {(1 - c[0], c[1]): entry for c, entry in stable_entries.items()}
        new_unstable_entries = {entry: (1 - c[0], c[1]) for entry, c in unstable_entries.items()}
        return (new_lines, new_stable_entries, new_unstable_entries)
    if nameup == ordering[1]:
        if nameleft == ordering[2]:
            c120 = np.cos(2 * np.pi / 3.0)
            s120 = np.sin(2 * np.pi / 3.0)
            new_lines = []
            for x, y in lines:
                newx = np.zeros_like(x)
                newy = np.zeros_like(y)
                for ii, xx in enumerate(x):
                    newx[ii] = c120 * (xx - cc[0]) - s120 * (y[ii] - cc[1]) + cc[0]
                    newy[ii] = s120 * (xx - cc[0]) + c120 * (y[ii] - cc[1]) + cc[1]
                new_lines.append([newx, newy])
            new_stable_entries = {(c120 * (c[0] - cc[0]) - s120 * (c[1] - cc[1]) + cc[0], s120 * (c[0] - cc[0]) + c120 * (c[1] - cc[1]) + cc[1]): entry for c, entry in stable_entries.items()}
            new_unstable_entries = {entry: (c120 * (c[0] - cc[0]) - s120 * (c[1] - cc[1]) + cc[0], s120 * (c[0] - cc[0]) + c120 * (c[1] - cc[1]) + cc[1]) for entry, c in unstable_entries.items()}
            return (new_lines, new_stable_entries, new_unstable_entries)
        c120 = np.cos(2 * np.pi / 3.0)
        s120 = np.sin(2 * np.pi / 3.0)
        new_lines = []
        for x, y in lines:
            newx = np.zeros_like(x)
            newy = np.zeros_like(y)
            for ii, xx in enumerate(x):
                newx[ii] = -c120 * (xx - 1.0) - s120 * y[ii] + 1.0
                newy[ii] = -s120 * (xx - 1.0) + c120 * y[ii]
            new_lines.append([newx, newy])
        new_stable_entries = {(-c120 * (c[0] - 1.0) - s120 * c[1] + 1.0, -s120 * (c[0] - 1.0) + c120 * c[1]): entry for c, entry in stable_entries.items()}
        new_unstable_entries = {entry: (-c120 * (c[0] - 1.0) - s120 * c[1] + 1.0, -s120 * (c[0] - 1.0) + c120 * c[1]) for entry, c in unstable_entries.items()}
        return (new_lines, new_stable_entries, new_unstable_entries)
    if nameup == ordering[2]:
        if nameleft == ordering[0]:
            c240 = np.cos(4 * np.pi / 3.0)
            s240 = np.sin(4 * np.pi / 3.0)
            new_lines = []
            for x, y in lines:
                newx = np.zeros_like(x)
                newy = np.zeros_like(y)
                for ii, xx in enumerate(x):
                    newx[ii] = c240 * (xx - cc[0]) - s240 * (y[ii] - cc[1]) + cc[0]
                    newy[ii] = s240 * (xx - cc[0]) + c240 * (y[ii] - cc[1]) + cc[1]
                new_lines.append([newx, newy])
            new_stable_entries = {(c240 * (c[0] - cc[0]) - s240 * (c[1] - cc[1]) + cc[0], s240 * (c[0] - cc[0]) + c240 * (c[1] - cc[1]) + cc[1]): entry for c, entry in stable_entries.items()}
            new_unstable_entries = {entry: (c240 * (c[0] - cc[0]) - s240 * (c[1] - cc[1]) + cc[0], s240 * (c[0] - cc[0]) + c240 * (c[1] - cc[1]) + cc[1]) for entry, c in unstable_entries.items()}
            return (new_lines, new_stable_entries, new_unstable_entries)
        c240 = np.cos(4 * np.pi / 3.0)
        s240 = np.sin(4 * np.pi / 3.0)
        new_lines = []
        for x, y in lines:
            newx = np.zeros_like(x)
            newy = np.zeros_like(y)
            for ii, xx in enumerate(x):
                newx[ii] = -c240 * xx - s240 * y[ii]
                newy[ii] = -s240 * xx + c240 * y[ii]
            new_lines.append([newx, newy])
        new_stable_entries = {(-c240 * c[0] - s240 * c[1], -s240 * c[0] + c240 * c[1]): entry for c, entry in stable_entries.items()}
        new_unstable_entries = {entry: (-c240 * c[0] - s240 * c[1], -s240 * c[0] + c240 * c[1]) for entry, c in unstable_entries.items()}
        return (new_lines, new_stable_entries, new_unstable_entries)
    raise ValueError('Invalid ordering.')