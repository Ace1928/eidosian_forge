import numpy as np
import os
import subprocess
import warnings
from ase.calculators.openmx.reader import rn as read_nth_to_last_value
def subplot_dos(self, axis, density=True, cum=False, pdos=False, atom_index=1, orbital='', spin='', erange=(-25, 20), fermi_level=True):
    """
        Plots a graph of (pseudo-)density of states against energy onto a given
        axis of a subplot.
        :param axis: matplotlib.pyplot.Axes object. This allows the graph to
                     plotted on any desired axis of a plot.
        :param density: If True, the density of states will be plotted
        :param cum: If True, the cumulative (or integrated) density of states
                    will be plotted
        :param pdos: If True, the pseudo-density of states will be plotted for
                     a given atom and orbital
        :param atom_index: If pdos is True, atom_index specifies which atom's
                           PDOS to plot.
        :param orbital: If pdos is True, orbital specifies which orbital's PDOS
                        to plot.
        :param spin: If '', density of states for both spin states will be
                     combined into one plot. If 'up' or 'down', a given spin
                     state's PDOS will be plotted.
        :return: None
        """
    p = ''
    bottom_index = 0
    atom_orbital = atom_orbital_spin = ''
    if pdos:
        p = 'p'
        atom_orbital += str(atom_index) + orbital
    atom_orbital_spin += atom_orbital + spin
    key = p + 'dos'
    density_color = 'r'
    cum_color = 'b'
    if spin == 'down':
        density_color = 'c'
        cum_color = 'm'
    if density and cum:
        axis_twin = axis.twinx()
        axis.plot(self.dos_dict[key + '_energies_' + atom_orbital], self.dos_dict[key + atom_orbital_spin], density_color)
        axis_twin.plot(self.dos_dict[key + '_energies_' + atom_orbital], self.dos_dict[key + '_cum_' + atom_orbital_spin], cum_color)
        max_density = max(self.dos_dict[key + atom_orbital_spin])
        max_cum = max(self.dos_dict[key + '_cum_' + atom_orbital_spin])
        if not max_density:
            max_density = 1.0
        if not max_cum:
            max_cum = 1
        axis.set_ylim(ymax=max_density)
        axis_twin.set_ylim(ymax=max_cum)
        axis.set_ylim(ymin=0.0)
        axis_twin.set_ylim(ymin=0.0)
        label_index = 0
        yticklabels = axis.get_yticklabels()
        if spin == 'down':
            bottom_index = len(yticklabels) - 1
        for t in yticklabels:
            if label_index == bottom_index or label_index == len(yticklabels) // 2:
                t.set_color(density_color)
            else:
                t.set_visible(False)
            label_index += 1
        label_index = 0
        yticklabels = axis_twin.get_yticklabels()
        if spin == 'down':
            bottom_index = len(yticklabels) - 1
        for t in yticklabels:
            if label_index == bottom_index or label_index == len(yticklabels) // 2:
                t.set_color(cum_color)
            else:
                t.set_visible(False)
            label_index += 1
        if spin == 'down':
            axis.set_ylim(axis.get_ylim()[::-1])
            axis_twin.set_ylim(axis_twin.get_ylim()[::-1])
    else:
        color = density_color
        if cum:
            color = cum_color
            key += '_cum_'
        key += atom_orbital_spin
        axis.plot(self.dos_dict[p + 'dos_energies_' + atom_orbital], self.dos_dict[key], color)
        maximum = max(self.dos_dict[key])
        if not maximum:
            maximum = 1.0
        axis.set_ylim(ymax=maximum)
        axis.set_ylim(ymin=0.0)
        label_index = 0
        yticklabels = axis.get_yticklabels()
        if spin == 'down':
            bottom_index = len(yticklabels) - 1
        for t in yticklabels:
            if label_index == bottom_index or label_index == len(yticklabels) // 2:
                t.set_color(color)
            else:
                t.set_visible(False)
            label_index += 1
        if spin == 'down':
            axis.set_ylim(axis.get_ylim()[::-1])
    if fermi_level:
        axis.axvspan(erange[0], 0.0, color='y', alpha=0.5)