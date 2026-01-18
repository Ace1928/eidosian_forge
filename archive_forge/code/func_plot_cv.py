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
@add_fig_kwargs
def plot_cv(self, tmin: float, tmax: float, ntemp: int, ylim: float | None=None, **kwargs) -> Figure:
    """Plots the constant volume specific heat C_v in a temperature range.

        Args:
            tmin: minimum temperature
            tmax: maximum temperature
            ntemp: number of steps
            ylim: tuple specifying the y-axis limits.
            kwargs: kwargs passed to the matplotlib function 'plot'.

        Returns:
            plt.figure: matplotlib figure
        """
    temperatures = np.linspace(tmin, tmax, ntemp)
    ylabel = '$C_v$ (J/K/mol)' if self.structure else '$C_v$ (J/K/mol-c)'
    return self._plot_thermo(self.dos.cv, temperatures, ylabel=ylabel, ylim=ylim, **kwargs)