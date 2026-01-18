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
def get_elt_projected_plots_color(self, zero_to_efermi=True, elt_ordered=None):
    """Returns a pyplot plot object with one plot where the band structure
        line color depends on the character of the band (along different
        elements). Each element is associated with red, green or blue
        and the corresponding rgb color depending on the character of the band
        is used. The method can only deal with binary and ternary compounds.

        Spin up and spin down are differentiated by a '-' and a '--' line.

        Args:
            zero_to_efermi: Automatically set the Fermi level as the plot's origin (i.e. subtract E - E_f).
                Defaults to True.
            elt_ordered: A list of Element ordered. The first one is red, second green, last blue.

        Raises:
            ValueError: if the number of elements is not 2 or 3.

        Returns:
            a pyplot object
        """
    band_linewidth = 3
    n_elems = len(self._bs.structure.elements)
    if n_elems > 3:
        raise ValueError(f'Can only plot binary and ternary compounds, got {n_elems} elements')
    if elt_ordered is None:
        elt_ordered = self._bs.structure.elements
    proj = self._get_projections_by_branches({e.symbol: ['s', 'p', 'd'] for e in self._bs.structure.elements})
    data = self.bs_plot_data(zero_to_efermi)
    ax = pretty_plot(12, 8)
    spins = [Spin.up]
    if self._bs.is_spin_polarized:
        spins = [Spin.up, Spin.down]
    self._make_ticks(ax)
    for spin in spins:
        for b in range(len(data['distances'])):
            for band_idx in range(self._nb_bands):
                for j in range(len(data['energy'][str(spin)][b][band_idx]) - 1):
                    sum_e = 0.0
                    for el in elt_ordered:
                        sum_e = sum_e + sum((proj[b][str(spin)][band_idx][j][str(el)][o] for o in proj[b][str(spin)][band_idx][j][str(el)]))
                    if sum_e == 0.0:
                        color = [0.0] * len(elt_ordered)
                    else:
                        color = [sum((proj[b][str(spin)][band_idx][j][str(el)][o] for o in proj[b][str(spin)][band_idx][j][str(el)])) / sum_e for el in elt_ordered]
                    if len(color) == 2:
                        color.append(0.0)
                        color[2] = color[1]
                        color[1] = 0.0
                    sign = '-'
                    if spin == Spin.down:
                        sign = '--'
                    ax.plot([data['distances'][b][j], data['distances'][b][j + 1]], [data['energy'][str(spin)][b][band_idx][j], data['energy'][str(spin)][b][band_idx][j + 1]], sign, color=color, linewidth=band_linewidth)
    if self._bs.is_metal():
        if zero_to_efermi:
            e_min = -10
            e_max = 10
            ax.set_ylim(e_min, e_max)
            ax.set_ylim(self._bs.efermi + e_min, self._bs.efermi + e_max)
    else:
        ax.set_ylim(data['vbm'][0][1] - 4.0, data['cbm'][0][1] + 2.0)
    x_max = data['distances'][-1][-1]
    ax.set_xlim(0, x_max)
    return ax