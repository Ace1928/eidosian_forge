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
def plot_seebeck_dop(self, temps='all', output='average'):
    """Plot the Seebeck in function of doping levels for different temperatures.

        Args:
            temps: the default 'all' plots all the temperatures in the analyzer.
                Specify a list of temperatures if you want to plot only some.
            output: with 'average' you get an average of the three directions
                with 'eigs' you get all the three directions.

        Returns:
            a matplotlib object
        """
    if output == 'average':
        sbk = self._bz.get_seebeck(output='average')
    elif output == 'eigs':
        sbk = self._bz.get_seebeck(output='eigs')
    tlist = sorted(sbk['n']) if temps == 'all' else temps
    ax = pretty_plot(22, 14)
    for i, dt in enumerate(['n', 'p']):
        plt.subplot(121 + i)
        for temp in tlist:
            if output == 'eigs':
                for xyz in range(3):
                    ax.semilogx(self._bz.doping[dt], list(zip(*sbk[dt][temp]))[xyz], marker='s', label=f'{xyz} {temp} K')
            elif output == 'average':
                ax.semilogx(self._bz.doping[dt], sbk[dt][temp], marker='s', label=f'{temp} K')
        ax.set_title(dt + '-type', fontsize=20)
        if i == 0:
            ax.set_ylabel('Seebeck coefficient ($\\mu$V/K)', fontsize=30.0)
        ax.set_xlabel('Doping concentration (cm$^{-3}$)', fontsize=30.0)
        p = 'lower right' if i == 0 else 'best'
        ax.legend(loc=p, fontsize=15)
        ax.grid()
        ax.set_xticks(fontsize=25)
        ax.set_yticks(fontsize=25)
    plt.tight_layout()
    return ax