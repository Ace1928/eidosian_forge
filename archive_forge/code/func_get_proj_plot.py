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
def get_proj_plot(self, site_comb: str | list[list[int]]='element', ylim: tuple[None | float, None | float] | None=None, units: Literal['thz', 'ev', 'mev', 'ha', 'cm-1', 'cm^-1']='thz', rgb_labels: tuple[None | str] | None=None) -> Axes:
    """Get a matplotlib object for the bandstructure plot projected along atomic
        sites.

        Args:
            site_comb: a list of list, for example, [[0],[1],[2,3,4]];
                the numbers in each sublist represents the indices of atoms;
                the atoms in a same sublist will be plotted in a same color;
                if not specified, unique elements are automatically grouped.
            ylim: Specify the y-axis (frequency) limits; by default None let
                the code choose.
            units: units for the frequencies. Accepted values thz, ev, mev, ha, cm-1, cm^-1.
                Defaults to "thz".
            rgb_labels: a list of rgb colors for the labels; if not specified,
                the colors will be automatically generated.
        """
    assert self._bs.structure is not None, 'Structure is required for get_proj_plot'
    elements = [elem.symbol for elem in self._bs.structure.elements]
    if site_comb == 'element':
        assert 2 <= len(elements) <= 4, 'the compound must have 2, 3 or 4 unique elements'
        indices: list[list[int]] = [[] for _ in range(len(elements))]
        for idx, elem in enumerate(self._bs.structure.species):
            for j, unique_species in enumerate(self._bs.structure.elements):
                if elem == unique_species:
                    indices[j].append(idx)
    else:
        assert isinstance(site_comb, list)
        assert 2 <= len(site_comb) <= 4, 'the length of site_comb must be 2, 3 or 4'
        all_sites = self._bs.structure.sites
        all_indices = {*range(len(all_sites))}
        for comb in site_comb:
            for idx in comb:
                assert 0 <= idx < len(all_sites), 'one or more indices in site_comb does not exist'
                all_indices.remove(idx)
        if len(all_indices) != 0:
            raise Exception(f'not all {len(all_sites)} indices are included in site_comb')
        indices = site_comb
    assert rgb_labels is None or len(rgb_labels) == len(indices), 'wrong number of rgb_labels'
    u = freq_units(units)
    _fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    self._make_ticks(ax)
    data = self.bs_plot_data()
    k_dist = np.array(data['distances']).flatten()
    for d in range(1, len(k_dist)):
        colors = []
        for idx in range(self.n_bands):
            eigenvec_1 = self._bs.eigendisplacements[idx][d - 1].flatten()
            eigenvec_2 = self._bs.eigendisplacements[idx][d].flatten()
            colors1 = self._get_weight(eigenvec_1, indices)
            colors2 = self._get_weight(eigenvec_2, indices)
            colors.append(self._make_color((colors1 + colors2) / 2))
        seg = np.zeros((self.n_bands, 2, 2))
        seg[:, :, 1] = self._bs.bands[:, d - 1:d + 1] * u.factor
        seg[:, 0, 0] = k_dist[d - 1]
        seg[:, 1, 0] = k_dist[d]
        ls = LineCollection(seg, colors=colors, linestyles='-', linewidths=2.5)
        ax.add_collection(ls)
    if ylim is None:
        y_max: float = max((max(b) for b in self._bs.bands)) * u.factor
        y_min: float = min((min(b) for b in self._bs.bands)) * u.factor
        y_margin = (y_max - y_min) * 0.05
        ylim = (y_min - y_margin, y_max + y_margin)
    ax.set_ylim(ylim)
    xlim = [min(k_dist), max(k_dist)]
    ax.set_xlim(xlim)
    ax.set_xlabel('$\\mathrm{Wave\\ Vector}$', fontsize=28)
    ylabel = f'$\\mathrm{{Frequencies\\ ({u.label})}}$'
    ax.set_ylabel(ylabel, fontsize=28)
    ax.tick_params(labelsize=28)
    labels: list[str]
    if rgb_labels is not None:
        labels = rgb_labels
    elif site_comb == 'element':
        labels = [elem.symbol for elem in self._bs.structure.elements]
    else:
        labels = [f'{idx}' for idx in range(len(site_comb))]
    if len(indices) == 2:
        BSDOSPlotter._rb_line(ax, labels[0], labels[1], 'best')
    elif len(indices) == 3:
        BSDOSPlotter._rgb_triangle(ax, labels[0], labels[1], labels[2], 'best')
    else:
        pass
    return ax