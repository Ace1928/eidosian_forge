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
@no_type_check
def get_elt_projected_plots(self, zero_to_efermi: bool=True, ylim=None, vbm_cbm_marker: bool=False) -> plt.Axes:
    """Method returning a plot composed of subplots along different elements.

        Returns:
            np.ndarray[plt.Axes]: 2x2 array of plt.Axes with different subfigures for each projection
                The blue and red colors are for spin up and spin down
                The bigger the red or blue dot in the band structure the higher
                character for the corresponding element and orbital
        """
    band_linewidth = 1.0
    proj = self._get_projections_by_branches({e.symbol: ['s', 'p', 'd'] for e in self._bs.structure.elements})
    data = self.bs_plot_data(zero_to_efermi)
    _fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    ax = pretty_plot(12, 8, ax=axs[0][0])
    e_min, e_max = (-4, 4)
    if self._bs.is_metal():
        e_min, e_max = (-10, 10)
    for idx, el in enumerate(self._bs.structure.elements, start=1):
        ax = plt.subplot(220 + idx)
        self._make_ticks(ax)
        for b in range(len(data['distances'])):
            for band_idx in range(self._nb_bands):
                ax.plot(data['distances'][b], data['energy'][str(Spin.up)][b][band_idx], '-', color=[192 / 255, 192 / 255, 192 / 255], linewidth=band_linewidth)
                if self._bs.is_spin_polarized:
                    ax.plot(data['distances'][b], data['energy'][str(Spin.down)][b][band_idx], '--', color=[128 / 255, 128 / 255, 128 / 255], linewidth=band_linewidth)
                    for j in range(len(data['energy'][str(Spin.up)][b][band_idx])):
                        markerscale = sum((proj[b][str(Spin.down)][band_idx][j][str(el)][o] for o in proj[b][str(Spin.down)][band_idx][j][str(el)]))
                        ax.plot(data['distances'][b][j], data['energy'][str(Spin.down)][b][band_idx][j], 'bo', markersize=markerscale * 15.0, color=[markerscale, 0.3 * markerscale, 0.4 * markerscale])
                for j in range(len(data['energy'][str(Spin.up)][b][band_idx])):
                    markerscale = sum((proj[b][str(Spin.up)][band_idx][j][str(el)][o] for o in proj[b][str(Spin.up)][band_idx][j][str(el)]))
                    ax.plot(data['distances'][b][j], data['energy'][str(Spin.up)][b][band_idx][j], 'o', markersize=markerscale * 15.0, color=[markerscale, 0.3 * markerscale, 0.4 * markerscale])
        if ylim is None:
            if self._bs.is_metal():
                if zero_to_efermi:
                    ax.set_ylim(e_min, e_max)
                else:
                    ax.set_ylim(self._bs.efermi + e_min, self._bs.efermi + e_max)
            else:
                if vbm_cbm_marker:
                    for cbm in data['cbm']:
                        ax.scatter(cbm[0], cbm[1], color='r', marker='o', s=100)
                    for vbm in data['vbm']:
                        ax.scatter(vbm[0], vbm[1], color='g', marker='o', s=100)
                ax.set_ylim(data['vbm'][0][1] + e_min, data['cbm'][0][1] + e_max)
        else:
            ax.set_ylim(ylim)
        ax.set_title(str(el))
    return axs