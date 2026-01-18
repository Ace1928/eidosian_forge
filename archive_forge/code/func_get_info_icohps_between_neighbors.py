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
def get_info_icohps_between_neighbors(self, isites=None, onlycation_isites=True):
    """
        Return infos about interactions between neighbors of a certain atom.

        Args:
            isites: list of site ids, if isite==None, all isites will be used
            onlycation_isites: will only use cations, if isite==None

        Returns:
            ICOHPNeighborsInfo
        """
    lowerlimit = self.lowerlimit
    upperlimit = self.upperlimit
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
    label_list = []
    atoms_list = []
    for isite in isites:
        for in_site, n_site in enumerate(self.list_neighsite[isite]):
            for in_site2, n_site2 in enumerate(self.list_neighsite[isite]):
                if in_site < in_site2:
                    unitcell1 = self._determine_unit_cell(n_site)
                    unitcell2 = self._determine_unit_cell(n_site2)
                    index_n_site = self._get_original_site(self.structure, n_site)
                    index_n_site2 = self._get_original_site(self.structure, n_site2)
                    if index_n_site < index_n_site2:
                        translation = list(np.array(unitcell1) - np.array(unitcell2))
                    elif index_n_site2 < index_n_site:
                        translation = list(np.array(unitcell2) - np.array(unitcell1))
                    else:
                        translation = list(np.array(unitcell1) - np.array(unitcell2))
                    icohps = self._get_icohps(icohpcollection=self.Icohpcollection, isite=index_n_site, lowerlimit=lowerlimit, upperlimit=upperlimit, only_bonds_to=self.only_bonds_to)
                    done = False
                    for icohp in icohps.values():
                        atomnr1 = self._get_atomnumber(icohp._atom1)
                        atomnr2 = self._get_atomnumber(icohp._atom2)
                        label = icohp._label
                        if index_n_site == atomnr1 and index_n_site2 == atomnr2 or (index_n_site == atomnr2 and index_n_site2 == atomnr1):
                            if atomnr1 != atomnr2:
                                if np.all(np.asarray(translation) == np.asarray(icohp._translation)):
                                    summed_icohps += icohp.summed_icohp
                                    list_icohps.append(icohp.summed_icohp)
                                    number_bonds += 1
                                    label_list.append(label)
                                    atoms_list.append([self.Icohpcollection._list_atom1[int(label) - 1], self.Icohpcollection._list_atom2[int(label) - 1]])
                            elif not done:
                                icohp_trans = -np.asarray([icohp._translation[0], icohp._translation[1], icohp._translation[2]])
                                if np.all(np.asarray(translation) == np.asarray(icohp._translation)) or np.all(np.asarray(translation) == icohp_trans):
                                    summed_icohps += icohp.summed_icohp
                                    list_icohps.append(icohp.summed_icohp)
                                    number_bonds += 1
                                    label_list.append(label)
                                    atoms_list.append([self.Icohpcollection._list_atom1[int(label) - 1], self.Icohpcollection._list_atom2[int(label) - 1]])
                                    done = True
    return ICOHPNeighborsInfo(summed_icohps, list_icohps, number_bonds, label_list, atoms_list, None)