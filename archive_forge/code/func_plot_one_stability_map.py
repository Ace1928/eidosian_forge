from __future__ import annotations
import copy
import itertools
import random
import warnings
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from sympy import Symbol
from sympy.solvers import linsolve, solve
from pymatgen.analysis.wulff import WulffShape
from pymatgen.core import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.surface import get_slab_regions
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.vasp.outputs import Locpot, Outcar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.due import Doi, due
from pymatgen.util.plotting import pretty_plot
def plot_one_stability_map(self, analyzer, max_r, delu_dict=None, label='', increments=50, delu_default=0, ax=None, from_sphere_area=False, e_units='keV', r_units='nanometers', normalize=False, scale_per_atom=False):
    """
        Returns the plot of the formation energy of a particle against its
            effect radius.

        Args:
            analyzer (SurfaceEnergyPlotter): Analyzer associated with the
                first polymorph
            max_r (float): The maximum radius of the particle to plot up to.
            delu_dict (dict): Dictionary of the chemical potentials to be set as
                constant. Note the key should be a sympy Symbol object of the
                format: Symbol("delu_el") where el is the name of the element.
            label (str): Label of the plot for legend
            increments (int): Number of plot points
            delu_default (float): Default value for all unset chemical potentials
            plt (pyplot): Plot
            from_sphere_area (bool): There are two ways to calculate the bulk
                formation energy. Either by treating the volume and thus surface
                area of the particle as a perfect sphere, or as a Wulff shape.
            r_units (str): Can be nanometers or Angstrom
            e_units (str): Can be keV or eV
            normalize (str): Whether or not to normalize energy by volume

        Returns:
            plt.Axes: matplotlib Axes object
        """
    ax = ax or pretty_plot(width=8, height=7)
    wulff_shape = analyzer.wulff_from_chempot(delu_dict=delu_dict, delu_default=delu_default, symprec=self.symprec)
    gform_list, r_list = ([], [])
    for radius in np.linspace(1e-06, max_r, increments):
        gform, radius = self.wulff_gform_and_r(wulff_shape, analyzer.ucell_entry, radius, from_sphere_area=from_sphere_area, r_units=r_units, e_units=e_units, normalize=normalize, scale_per_atom=scale_per_atom)
        gform_list.append(gform)
        r_list.append(radius)
    ru = 'nm' if r_units == 'nanometers' else '\\AA'
    ax.xlabel(f'Particle radius (${ru}$)')
    eu = f'${e_units}/{ru}^3$'
    ax.ylabel(f'$G_{{form}}$ ({eu})')
    ax.plot(r_list, gform_list, label=label)
    return ax