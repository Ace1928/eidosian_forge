from __future__ import annotations
import logging
from collections import namedtuple
from typing import TYPE_CHECKING, Callable
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from matplotlib.collections import LineCollection
from monty.json import jsanitize
from pymatgen.electronic_structure.plotter import BSDOSPlotter, plot_brillouin_zone
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.gruneisen import GruneisenPhononBandStructureSymmLine
from pymatgen.util.plotting import add_fig_kwargs, get_ax_fig, pretty_plot
def plot_compare_gs(self, other_plotter: GruneisenPhononBSPlotter) -> Axes:
    """Plot two band structure for comparison. One is in red the other in blue.
        The two band structures need to be defined on the same symmetry lines!
        and the distance between symmetry lines is
        the one of the band structure used to build the PhononBSPlotter.

        Args:
            other_plotter (GruneisenPhononBSPlotter): another phonon DOS plotter defined along
                the same symmetry lines.

        Raises:
            ValueError: if the two plotters are incompatible (due to different data lengths)

        Returns:
            a matplotlib object with both band structures
        """
    data_orig = self.bs_plot_data()
    data = other_plotter.bs_plot_data()
    len_orig = len(data_orig['distances'])
    len_other = len(data['distances'])
    if len_orig != len_other:
        raise ValueError(f'The two plotters are incompatible, plotting data have different lengths ({len_orig} vs {len_other}).')
    ax = self.get_plot()
    band_linewidth = 1
    for band_idx in range(other_plotter.n_bands):
        for dist_idx in range(len(data_orig['distances'])):
            ax.plot(data_orig['distances'][dist_idx], [data['gruneisen'][dist_idx][band_idx][j] for j in range(len(data_orig['distances'][dist_idx]))], 'r-', linewidth=band_linewidth)
    return ax