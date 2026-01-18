from __future__ import annotations
import itertools
import logging
import re
import warnings
from copy import deepcopy
from functools import cmp_to_key, partial
from multiprocessing import Pool
from typing import TYPE_CHECKING, Any, no_type_check
import numpy as np
from monty.json import MontyDecoder, MSONable
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.special import comb
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.analysis.reaction_calculator import Reaction, ReactionError
from pymatgen.core import Composition, Element
from pymatgen.core.ion import Ion
from pymatgen.entries.compatibility import MU_H2O
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.util.coord import Simplex
from pymatgen.util.due import Doi, due
from pymatgen.util.plotting import pretty_plot
from pymatgen.util.string import Stringify
@no_type_check
def get_pourbaix_plot(self, limits: tuple[float, float] | None=None, title: str='', label_domains: bool=True, label_fontsize: int=20, show_water_lines: bool=True, show_neutral_axes: bool=True, ax: plt.Axes=None) -> plt.Axes:
    """
        Plot Pourbaix diagram.

        Args:
            limits: 2D list containing limits of the Pourbaix diagram
                of the form [[xlo, xhi], [ylo, yhi]]
            title (str): Title to display on plot
            label_domains (bool): whether to label Pourbaix domains
            label_fontsize: font size for domain labels
            show_water_lines: whether to show dashed lines indicating the region
                of water stability.
            show_neutral_axes; whether to show dashed horizontal and vertical lines
                at 0 V and pH 7, respectively.
            ax (Axes): Matplotlib Axes instance for plotting

        Returns:
            Axes: matplotlib Axes object with Pourbaix diagram
        """
    if limits is None:
        limits = [[-2, 16], [-3, 3]]
    ax = ax or pretty_plot(16)
    xlim, ylim = limits
    lw = 3
    if show_water_lines:
        h_line = np.transpose([[xlim[0], -xlim[0] * PREFAC], [xlim[1], -xlim[1] * PREFAC]])
        o_line = np.transpose([[xlim[0], -xlim[0] * PREFAC + 1.23], [xlim[1], -xlim[1] * PREFAC + 1.23]])
        ax.plot(h_line[0], h_line[1], 'r--', linewidth=lw)
        ax.plot(o_line[0], o_line[1], 'r--', linewidth=lw)
    if show_neutral_axes:
        neutral_line = np.transpose([[7, ylim[0]], [7, ylim[1]]])
        V0_line = np.transpose([[xlim[0], 0], [xlim[1], 0]])
        ax.plot(neutral_line[0], neutral_line[1], 'k-.', linewidth=lw)
        ax.plot(V0_line[0], V0_line[1], 'k-.', linewidth=lw)
    for entry, vertices in self._pbx._stable_domain_vertices.items():
        center = np.average(vertices, axis=0)
        x, y = np.transpose(np.vstack([vertices, vertices[0]]))
        ax.plot(x, y, 'k-', linewidth=lw)
        if label_domains:
            ax.annotate(generate_entry_label(entry), center, ha='center', va='center', fontsize=label_fontsize, color='b').draggable()
    ax.set_title(title, fontsize=20, fontweight='bold')
    ax.set(xlabel='pH', ylabel='E (V)', xlim=xlim, ylim=ylim)
    return ax