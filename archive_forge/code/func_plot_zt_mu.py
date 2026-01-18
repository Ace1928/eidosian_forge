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
def plot_zt_mu(self, temp: float=600, output: str='eig', relaxation_time: float=1e-14, xlim: Sequence[float] | None=None) -> plt.Axes:
    """Plot the ZT as function of Fermi level.

        Args:
            temp (float): the temperature
            output (str): "eig" or "average"
            relaxation_time (float): A relaxation time in s. Defaults to 1e-14 and the plot is in
               units of relaxation time
            xlim (tuple[float, float]): a 2-tuple of min and max fermi energy. Defaults to (0, band gap)

        Returns:
            plt.Axes: matplotlib axes object
        """
    ax = pretty_plot(9, 7)
    zt = self._bz.get_zt(relaxation_time=relaxation_time, output=output, doping_levels=False)[temp]
    ax.plot(self._bz.mu_steps, zt, linewidth=3.0)
    self._plot_bg_limits(ax)
    self._plot_doping(ax, temp)
    if output == 'eig':
        ax.legend(['ZT$_1$', 'ZT$_2$', 'ZT$_3$'])
    if xlim is None:
        ax.set_xlim(-0.5, self._bz.gap + 0.5)
    else:
        ax.set_xlim(xlim)
    ax.set_ylabel('ZT', fontsize=30.0)
    ax.set_xlabel('E-E$_f$ (eV)', fontsize=30.0)
    ax.set_xticks(fontsize=25)
    ax.set_yticks(fontsize=25)
    plt.tight_layout()
    return ax