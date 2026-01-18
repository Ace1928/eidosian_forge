from __future__ import annotations
import collections
import copy
import math
import tempfile
from typing import TYPE_CHECKING, NamedTuple
import numpy as np
from monty.dev import deprecated
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments
from pymatgen.analysis.local_env import NearNeighbors
from pymatgen.electronic_structure.cohp import CompleteCohp
from pymatgen.electronic_structure.core import Spin
from pymatgen.electronic_structure.plotter import CohpPlotter
from pymatgen.io.lobster import Charge, Icohplist
from pymatgen.util.due import Doi, due
def plot_cohps_of_neighbors(self, path_to_cohpcar: str | None='COHPCAR.lobster', obj_cohpcar: CompleteCohp | None=None, isites: list[int] | None=None, onlycation_isites: bool=True, only_bonds_to: list[str] | None=None, per_bond: bool=False, summed_spin_channels: bool=False, xlim=None, ylim=(-10, 6), integrated: bool=False):
    """
        Will plot summed cohps or cobis or coops
        (please be careful in the spin polarized case (plots might overlap (exactly!)).

        Args:
            path_to_cohpcar: str, path to COHPCAR or COOPCAR or COBICAR
            obj_cohpcar: CompleteCohp object
            isites: list of site ids, if isite==[], all isites will be used to add the icohps of the neighbors
            onlycation_isites: bool, will only use cations, if isite==[]
            only_bonds_to: list of str, only anions in this list will be considered
            per_bond: bool, will lead to a normalization of the plotted COHP per number of bond if True,
            otherwise the sum
            will be plotted
            xlim: list of float, limits of x values
            ylim: list of float, limits of y values
            integrated: bool, if true will show integrated cohp instead of cohp

        Returns:
            plt of the cohps or coops or cobis
        """
    cp = CohpPlotter(are_cobis=self.are_cobis, are_coops=self.are_coops)
    plotlabel, summed_cohp = self.get_info_cohps_to_neighbors(path_to_cohpcar, obj_cohpcar, isites, only_bonds_to, onlycation_isites, per_bond, summed_spin_channels=summed_spin_channels)
    cp.add_cohp(plotlabel, summed_cohp)
    ax = cp.get_plot(integrated=integrated)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    return ax