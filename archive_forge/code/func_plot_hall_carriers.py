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
def plot_hall_carriers(self, temp=300):
    """Plot the Hall carrier concentration in function of Fermi level.

        Args:
            temp: the temperature

        Returns:
            a matplotlib object
        """
    ax = pretty_plot(9, 7)
    hall_carriers = [abs(i) for i in self._bz.get_hall_carrier_concentration()[temp]]
    ax.semilogy(self._bz.mu_steps, hall_carriers, linewidth=3.0, color='r')
    self._plot_bg_limits(ax)
    self._plot_doping(ax, temp)
    ax.set_xlim(-0.5, self._bz.gap + 0.5)
    ax.set_ylim(100000000000000.0, 1e+22)
    ax.set_ylabel('Hall carrier concentration (cm-3)', fontsize=30.0)
    ax.set_xlabel('E-E$_f$ (eV)', fontsize=30)
    ax.set_xticks(fontsize=25)
    ax.set_yticks(fontsize=25)
    plt.tight_layout()
    return ax