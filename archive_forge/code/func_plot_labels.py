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
def plot_labels(labels, lattice=None, coords_are_cartesian=False, ax: plt.Axes=None, **kwargs):
    """Adds labels to a matplotlib Axes.

    Args:
        labels: dict containing the label as a key and the coordinates as value.
        lattice: Lattice object used to convert from reciprocal to Cartesian coordinates
        coords_are_cartesian: Set to True if you are providing.
            coordinates in Cartesian coordinates. Defaults to False.
            Requires lattice if False.
        ax: matplotlib Axes or None if a new figure should be created.
        kwargs: kwargs passed to the matplotlib function 'text'. Color defaults to blue
            and size to 25.

    Returns:
        matplotlib figure and matplotlib ax
    """
    ax, fig = get_ax3d_fig(ax)
    if 'color' not in kwargs:
        kwargs['color'] = 'b'
    if 'size' not in kwargs:
        kwargs['size'] = 25
    for k, coords in labels.items():
        label = k
        if k.startswith('\\') or k.find('_') != -1:
            label = f'${k}$'
        off = 0.01
        if coords_are_cartesian:
            coords = np.array(coords)
        else:
            if lattice is None:
                raise ValueError('coords_are_cartesian False requires the lattice')
            coords = lattice.get_cartesian_coords(coords)
        ax.text(*coords + off, s=label, **kwargs)
    return (fig, ax)