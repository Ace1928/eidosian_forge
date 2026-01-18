from __future__ import annotations
import logging
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING
import numpy as np
from scipy.optimize import leastsq, minimize
from pymatgen.core.units import FloatWithUnit
from pymatgen.util.plotting import add_fig_kwargs, get_ax_fig, pretty_plot
@add_fig_kwargs
def plot_ax(self, ax: plt.Axes=None, fontsize=12, **kwargs):
    """
        Plot the equation of state on axis `ax`.

        Args:
            ax: matplotlib Axes or None if a new figure should be created.
            fontsize: Legend fontsize.
            color (str): plot color.
            label (str): Plot label
            text (str): Legend text (options)

        Returns:
            plt.Figure: matplotlib figure.
        """
    ax, fig = get_ax_fig(ax=ax)
    color = kwargs.get('color', 'r')
    label = kwargs.get('label', f'{type(self).__name__} fit')
    lines = [f'Equation of State: {type(self).__name__}', f'Minimum energy = {self.e0:1.2f} eV', f'Minimum or reference volume = {self.v0:1.2f} Ang^3', f'Bulk modulus = {self.b0:1.2f} eV/Ang^3 = {self.b0_GPa:1.2f} GPa', f'Derivative of bulk modulus w.r.t. pressure = {self.b1:1.2f}']
    text = '\n'.join(lines)
    text = kwargs.get('text', text)
    ax.plot(self.volumes, self.energies, linestyle='None', marker='o', color=color)
    vmin, vmax = (min(self.volumes), max(self.volumes))
    vmin, vmax = (vmin - 0.01 * abs(vmin), vmax + 0.01 * abs(vmax))
    vfit = np.linspace(vmin, vmax, 100)
    ax.plot(vfit, self.func(vfit), linestyle='dashed', color=color, label=label)
    ax.grid(visible=True)
    ax.set_xlabel('Volume $\\AA^3$')
    ax.set_ylabel('Energy (eV)')
    ax.legend(loc='best', shadow=True)
    ax.text(0.5, 0.5, text, fontsize=fontsize, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    return fig