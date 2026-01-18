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
def plot_thermodynamic_properties(self, tmin: float, tmax: float, ntemp: int, ylim: float | None=None, **kwargs) -> Figure:
    """Plots all the thermodynamic properties in a temperature range.

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
    mol = '' if self.structure else '-c'
    fig = self._plot_thermo(self.dos.cv, temperatures, ylabel='Thermodynamic properties', ylim=ylim, label=f'$C_v$ (J/K/mol{mol})', **kwargs)
    self._plot_thermo(self.dos.entropy, temperatures, ylim=ylim, ax=fig.axes[0], label=f'$S$ (J/K/mol{mol})', **kwargs)
    self._plot_thermo(self.dos.internal_energy, temperatures, ylim=ylim, ax=fig.axes[0], factor=0.001, label=f'$\\Delta E$ (kJ/mol{mol})', **kwargs)
    self._plot_thermo(self.dos.helmholtz_free_energy, temperatures, ylim=ylim, ax=fig.axes[0], factor=0.001, label=f'$\\Delta F$ (kJ/mol{mol})', **kwargs)
    fig.axes[0].legend(loc='best')
    return fig