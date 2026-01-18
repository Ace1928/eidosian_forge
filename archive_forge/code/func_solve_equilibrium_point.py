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
def solve_equilibrium_point(self, analyzer1, analyzer2, delu_dict=None, delu_default=0, units='nanometers'):
    """
        Gives the radial size of two particles where equilibrium is reached
            between both particles. NOTE: the solution here is not the same
            as the solution visualized in the plot because solving for r
            requires that both the total surface area and volume of the
            particles are functions of r.

        Args:
            analyzer1 (SurfaceEnergyPlotter): Analyzer associated with the
                first polymorph
            analyzer2 (SurfaceEnergyPlotter): Analyzer associated with the
                second polymorph
            delu_dict (dict): Dictionary of the chemical potentials to be set as
                constant. Note the key should be a sympy Symbol object of the
                format: Symbol("delu_el") where el is the name of the element.
            delu_default (float): Default value for all unset chemical potentials
            units (str): Can be nanometers or Angstrom

        Returns:
            Particle radius in nm
        """
    wulff1 = analyzer1.wulff_from_chempot(delu_dict=delu_dict or {}, delu_default=delu_default, symprec=self.symprec)
    wulff2 = analyzer2.wulff_from_chempot(delu_dict=delu_dict or {}, delu_default=delu_default, symprec=self.symprec)
    delta_gamma = wulff1.weighted_surface_energy - wulff2.weighted_surface_energy
    delta_E = self.bulk_gform(analyzer1.ucell_entry) - self.bulk_gform(analyzer2.ucell_entry)
    radius = -3 * delta_gamma / delta_E
    return radius / 10 if units == 'nanometers' else radius