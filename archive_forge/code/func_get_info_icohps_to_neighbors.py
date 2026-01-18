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
def get_info_icohps_to_neighbors(self, isites=None, onlycation_isites=True):
    """
        This method returns information on the icohps of neighbors for certain sites as identified by their site id.
        This is useful for plotting the relevant cohps of a site in the structure.
        (could be ICOOPLIST.lobster or ICOHPLIST.lobster or ICOBILIST.lobster)

        Args:
            isites: list of site ids. If isite==None, all isites will be used to add the icohps of the neighbors
            onlycation_isites: if True and if isite==None, it will only analyse the sites of the cations

        Returns:
            ICOHPNeighborsInfo
        """
    if self.valences is None and onlycation_isites:
        raise ValueError('No valences are provided')
    if isites is None:
        if onlycation_isites:
            isites = [i for i in range(len(self.structure)) if self.valences[i] >= 0.0]
        else:
            isites = list(range(len(self.structure)))
    summed_icohps = 0.0
    list_icohps = []
    number_bonds = 0
    labels = []
    atoms = []
    final_isites = []
    for ival, _site in enumerate(self.structure):
        if ival in isites:
            for keys, icohpsum in zip(self.list_keys[ival], self.list_icohps[ival]):
                summed_icohps += icohpsum
                list_icohps.append(icohpsum)
                labels.append(keys)
                atoms.append([self.Icohpcollection._list_atom1[int(keys) - 1], self.Icohpcollection._list_atom2[int(keys) - 1]])
                number_bonds += 1
                final_isites.append(ival)
    return ICOHPNeighborsInfo(summed_icohps, list_icohps, number_bonds, labels, atoms, final_isites)