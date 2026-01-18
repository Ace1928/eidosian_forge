import numpy as np
import os
import subprocess
import warnings
from ase.calculators.openmx.reader import rn as read_nth_to_last_value
def plot_dos(self, density=True, cum=False, pdos=False, orbital_list=None, atom_index_list=None, spins=('up', 'down'), fermi_level=True, spin_polarization=False, erange=(-25, 20), atoms=None, method='Tetrahedron', file_format=None):
    """
        Generates a graphical figure containing possible subplots of different
        PDOSs of different atoms, orbitals and spin state combinations.
        :param density: If True, density of states will be plotted
        :param cum: If True, cumulative density of states will be plotted
        :param pdos: If True, pseudo-density of states will be plotted for
                     given atoms and orbitals
        :param atom_index_list: If pdos is True, atom_index_list specifies
                                which atoms will have their PDOS plotted.
        :param orbital_list: If pdos is True, orbital_list specifies which
                             orbitals will have their PDOS plotted.
        :param spins: If '' in spins, density of states for both spin states
                      will be combined into one graph. If 'up' or
        'down' in spins, a given spin state's PDOS graph will be plotted.
        :param spin_polarization: If spin_polarization is False then spin
                                  states will not be separated in different
                                  PDOS's.
        :param erange: range of energies to view DOS
        :return: matplotlib.figure.Figure and matplotlib.axes.Axes object
        """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    if not spin_polarization:
        spins = ['']
    number_of_spins = len(spins)
    if orbital_list is None:
        orbital_list = ['']
    number_of_atoms = 1
    number_of_orbitals = 1
    p = ''
    if pdos:
        p = 'P'
        if atom_index_list is None:
            atom_index_list = [i + 1 for i in range(len(atoms))]
        number_of_atoms = len(atom_index_list)
        number_of_orbitals = len(orbital_list)
    figure, axes = plt.subplots(number_of_orbitals * number_of_spins, number_of_atoms, sharex=True, sharey=False, squeeze=False)
    for i in range(number_of_orbitals):
        for s in range(number_of_spins):
            row_index = i * number_of_spins + s
            for j in range(number_of_atoms):
                self.subplot_dos(fermi_level=fermi_level, density=density, axis=axes[row_index][j], erange=erange, atom_index=atom_index_list[j], pdos=pdos, orbital=orbital_list[i], spin=spins[s], cum=cum)
                if j == 0 and pdos:
                    orbital = orbital_list[i]
                    if orbital == '':
                        orbital = 'All'
                    if spins[s]:
                        orbital += ' ' + spins[s]
                    axes[row_index][j].set_ylabel(orbital)
                if row_index == 0 and pdos:
                    atom_symbol = ''
                    if atoms:
                        atom_symbol = ' (' + atoms[atom_index_list[j]].symbol + ')'
                    axes[row_index][j].set_title('Atom ' + str(atom_index_list[j]) + atom_symbol)
                if row_index == number_of_orbitals * number_of_spins - 1:
                    axes[row_index][j].set_xlabel('Energy above Fermi Level (eV)')
    plt.xlim(xmin=erange[0], xmax=erange[1])
    if density and cum:
        figure.suptitle(self.calc.label)
        xdata = (0.0, 1.0)
        ydata = (0.0, 0.0)
        key_tuple = (Line2D(color='r', xdata=xdata, ydata=ydata), Line2D(color='b', xdata=xdata, ydata=ydata))
        if spin_polarization:
            key_tuple = (Line2D(color='r', xdata=xdata, ydata=ydata), Line2D(color='b', xdata=xdata, ydata=ydata), Line2D(color='c', xdata=xdata, ydata=ydata), Line2D(color='m', xdata=xdata, ydata=ydata))
        title_tuple = (p + 'DOS (eV^-1)', 'Number of States per Unit Cell')
        if spin_polarization:
            title_tuple = (p + 'DOS (eV^-1), spin up', 'Number of States per Unit Cell, spin up', p + 'DOS (eV^-1), spin down', 'Number of States per Unit Cell, spin down')
        figure.legend(key_tuple, title_tuple, 'lower center')
    elif density:
        figure.suptitle(self.calc.prefix + ': ' + p + 'DOS (eV^-1)')
    elif cum:
        figure.suptitle(self.calc.prefix + ': Number of States')
    extra_margin = 0
    if density and cum and spin_polarization:
        extra_margin = 0.1
    plt.subplots_adjust(hspace=0.0, bottom=0.2 + extra_margin, wspace=0.29, left=0.09, right=0.95)
    if file_format:
        orbitals = ''
        if pdos:
            atom_index_list = map(str, atom_index_list)
            atoms = '&'.join(atom_index_list)
            if '' in orbital_list:
                all_index = orbital_list.index('')
                orbital_list.remove('')
                orbital_list.insert(all_index, 'all')
            orbitals = ''.join(orbital_list)
        plt.savefig(filename=self.calc.label + '.' + p + 'DOS.' + method + '.atoms' + atoms + '.' + orbitals + '.' + file_format)
    if not file_format:
        plt.show()
    return (figure, axes)