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
def get_chempot_range_map_plot(self, elements, referenced=True):
    """
        Returns a plot of the chemical potential range _map. Currently works
        only for 3-component PDs.

        Note: this functionality is now included in the ChemicalPotentialDiagram
        class (pymatgen.analysis.chempot_diagram).

        Args:
            elements: Sequence of elements to be considered as independent
                variables. E.g., if you want to show the stability ranges of
                all Li-Co-O phases w.r.t. to uLi and uO, you will supply
                [Element("Li"), Element("O")]
            referenced: if True, gives the results with a reference being the
                energy of the elemental phase. If False, gives absolute values.

        Returns:
            plt.Axes: matplotlib axes object.
        """
    ax = pretty_plot(12, 8)
    chempot_ranges = self._pd.get_chempot_range_map(elements, referenced=referenced)
    missing_lines = {}
    excluded_region = []
    for entry, lines in chempot_ranges.items():
        comp = entry.composition
        center_x = 0
        center_y = 0
        coords = []
        contain_zero = any((comp.get_atomic_fraction(el) == 0 for el in elements))
        is_boundary = not contain_zero and sum((comp.get_atomic_fraction(el) for el in elements)) == 1
        for line in lines:
            x, y = line.coords.transpose()
            plt.plot(x, y, 'k-')
            for coord in line.coords:
                if not in_coord_list(coords, coord):
                    coords.append(coord.tolist())
                    center_x += coord[0]
                    center_y += coord[1]
            if is_boundary:
                excluded_region.extend(line.coords)
        if coords and contain_zero:
            missing_lines[entry] = coords
        else:
            xy = (center_x / len(coords), center_y / len(coords))
            plt.annotate(latexify(entry.name), xy, fontsize=22)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    excluded_region.append([xlim[1], ylim[1]])
    excluded_region = sorted(excluded_region, key=lambda c: c[0])
    x, y = np.transpose(excluded_region)
    plt.fill(x, y, '0.80')
    el0 = elements[0]
    el1 = elements[1]
    for entry, coords in missing_lines.items():
        center_x = sum((c[0] for c in coords))
        center_y = sum((c[1] for c in coords))
        comp = entry.composition
        is_x = comp.get_atomic_fraction(el0) < 0.01
        is_y = comp.get_atomic_fraction(el1) < 0.01
        n = len(coords)
        if not (is_x and is_y):
            if is_x:
                coords = sorted(coords, key=lambda c: c[1])
                for idx in [0, -1]:
                    x = [min(xlim), coords[idx][0]]
                    y = [coords[idx][1], coords[idx][1]]
                    plt.plot(x, y, 'k')
                    center_x += min(xlim)
                    center_y += coords[idx][1]
            elif is_y:
                coords = sorted(coords, key=lambda c: c[0])
                for idx in [0, -1]:
                    x = [coords[idx][0], coords[idx][0]]
                    y = [coords[idx][1], min(ylim)]
                    plt.plot(x, y, 'k')
                    center_x += coords[idx][0]
                    center_y += min(ylim)
            xy = (center_x / (n + 2), center_y / (n + 2))
        else:
            center_x = sum((coord[0] for coord in coords)) + xlim[0]
            center_y = sum((coord[1] for coord in coords)) + ylim[0]
            xy = (center_x / (n + 1), center_y / (n + 1))
        ax.annotate(latexify(entry.name), xy, horizontalalignment='center', verticalalignment='center', fontsize=22)
    ax.set_xlabel(f'$\\mu_{{{el0.symbol}}} - \\mu_{{{el0.symbol}}}^0$ (eV)')
    ax.set_ylabel(f'$\\mu_{{{el1.symbol}}} - \\mu_{{{el1.symbol}}}^0$ (eV)')
    plt.tight_layout()
    return ax