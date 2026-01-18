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
@add_fig_kwargs
def plot_brillouin_zone(bz_lattice, lines=None, labels=None, kpoints=None, fold=False, coords_are_cartesian: bool=False, ax: plt.Axes=None, **kwargs):
    """Plots a 3D representation of the Brillouin zone of the structure.
    Can add to the plot paths, labels and kpoints.

    Args:
        bz_lattice: Lattice object of the Brillouin zone
        lines: list of lists of coordinates. Each list represent a different path
        labels: dict containing the label as a key and the coordinates as value.
        kpoints: list of coordinates
        fold: whether the points should be folded inside the first Brillouin Zone.
            Defaults to False. Requires lattice if True.
        coords_are_cartesian: Set to True if you are providing
            coordinates in Cartesian coordinates. Defaults to False.
        ax: matplotlib Axes or None if a new figure should be created.
        kwargs: provided by add_fig_kwargs decorator

    Returns:
        matplotlib figure
    """
    fig, ax = plot_lattice_vectors(bz_lattice, ax=ax)
    plot_wigner_seitz(bz_lattice, ax=ax)
    if lines is not None:
        for line in lines:
            plot_path(line, bz_lattice, coords_are_cartesian=coords_are_cartesian, ax=ax)
    if labels is not None:
        plot_labels(labels, bz_lattice, coords_are_cartesian=coords_are_cartesian, ax=ax)
        plot_points(labels.values(), bz_lattice, coords_are_cartesian=coords_are_cartesian, fold=False, ax=ax)
    if kpoints is not None:
        plot_points(kpoints, bz_lattice, coords_are_cartesian=coords_are_cartesian, ax=ax, fold=fold)
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)
    ax.axis('off')
    return fig