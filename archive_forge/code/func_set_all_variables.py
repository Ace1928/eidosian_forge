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
def set_all_variables(self, delu_dict, delu_default):
    """
        Sets all chemical potential values and returns a dictionary where
            the key is a sympy Symbol and the value is a float (chempot).

        Args:
            entry (SlabEntry): Computed structure entry of the slab
            delu_dict (dict): Dictionary of the chemical potentials to be set as
                constant. Note the key should be a sympy Symbol object of the
                format: Symbol("delu_el") where el is the name of the element.
            delu_default (float): Default value for all unset chemical potentials

        Returns:
            Dictionary of set chemical potential values
        """
    all_delu_dict = {}
    for du in self.list_of_chempots:
        if delu_dict and du in delu_dict:
            all_delu_dict[du] = delu_dict[du]
        elif du == 1:
            all_delu_dict[du] = du
        else:
            all_delu_dict[du] = delu_default
    return all_delu_dict