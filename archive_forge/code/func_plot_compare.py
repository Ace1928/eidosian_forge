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
def plot_compare(self, other_plotter, legend=True) -> plt.Axes:
    """Plot two band structure for comparison. One is in red the other in blue
        (no difference in spins). The two band structures need to be defined
        on the same symmetry lines! and the distance between symmetry lines is
        the one of the band structure used to build the BSPlotter.

        Args:
            other_plotter: Another band structure object defined along the same symmetry lines
            legend: True to add a legend to the plot

        Returns:
            plt.Axes: matplotlib Axes object with both band structures
        """
    warnings.warn('Deprecated method. Use BSPlotter([sbs1,sbs2,...]).get_plot() instead.')
    ax = self.get_plot()
    data_orig = self.bs_plot_data()
    data = other_plotter.bs_plot_data()
    band_linewidth = 1
    for i in range(other_plotter._nb_bands):
        for d in range(len(data_orig['distances'])):
            ax.plot(data_orig['distances'][d], [e[str(Spin.up)][i] for e in data['energy']][d], 'c-', linewidth=band_linewidth)
            if other_plotter._bs.is_spin_polarized:
                ax.plot(data_orig['distances'][d], [e[str(Spin.down)][i] for e in data['energy']][d], 'm--', linewidth=band_linewidth)
    if legend:
        handles = [mlines.Line2D([], [], linewidth=2, color='b', label='bs 1 up'), mlines.Line2D([], [], linewidth=2, color='r', label='bs 1 down', linestyle='--'), mlines.Line2D([], [], linewidth=2, color='c', label='bs 2 up'), mlines.Line2D([], [], linewidth=2, color='m', linestyle='--', label='bs 2 down')]
        ax.legend(handles=handles)
    return ax