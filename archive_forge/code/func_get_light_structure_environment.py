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
def get_light_structure_environment(self, only_cation_environments=False, only_indices=None):
    """
        Return a LobsterLightStructureEnvironments object
        if the structure only contains coordination environments smaller 13.

        Args:
            only_cation_environments: only data for cations will be returned
            only_indices: will only evaluate the list of isites in this list

        Returns:
            LobsterLightStructureEnvironments
        """
    lgf = LocalGeometryFinder()
    lgf.setup_structure(structure=self.structure)
    list_ce_symbols = []
    list_csm = []
    list_permut = []
    for ival, _neigh_coords in enumerate(self.list_coords):
        if len(_neigh_coords) > 13:
            raise ValueError('Environment cannot be determined. Number of neighbors is larger than 13.')
        if _neigh_coords != []:
            lgf.setup_local_geometry(isite=ival, coords=_neigh_coords, optimization=2)
            cncgsm = lgf.get_coordination_symmetry_measures(optimization=2)
            list_ce_symbols.append(min(cncgsm.items(), key=lambda t: t[1]['csm_wcs_ctwcc'])[0])
            list_csm.append(min(cncgsm.items(), key=lambda t: t[1]['csm_wcs_ctwcc'])[1]['csm_wcs_ctwcc'])
            list_permut.append(min(cncgsm.items(), key=lambda t: t[1]['csm_wcs_ctwcc'])[1]['indices'])
        else:
            list_ce_symbols.append(None)
            list_csm.append(None)
            list_permut.append(None)
    if only_indices is None:
        if not only_cation_environments:
            lse = LobsterLightStructureEnvironments.from_Lobster(list_ce_symbol=list_ce_symbols, list_csm=list_csm, list_permutation=list_permut, list_neighsite=self.list_neighsite, list_neighisite=self.list_neighisite, structure=self.structure, valences=self.valences)
        else:
            new_list_ce_symbols = []
            new_list_csm = []
            new_list_permut = []
            new_list_neighsite = []
            new_list_neighisite = []
            for ival, val in enumerate(self.valences):
                if val >= 0.0:
                    new_list_ce_symbols.append(list_ce_symbols[ival])
                    new_list_csm.append(list_csm[ival])
                    new_list_permut.append(list_permut[ival])
                    new_list_neighisite.append(self.list_neighisite[ival])
                    new_list_neighsite.append(self.list_neighsite[ival])
                else:
                    new_list_ce_symbols.append(None)
                    new_list_csm.append(None)
                    new_list_permut.append([])
                    new_list_neighisite.append([])
                    new_list_neighsite.append([])
            lse = LobsterLightStructureEnvironments.from_Lobster(list_ce_symbol=new_list_ce_symbols, list_csm=new_list_csm, list_permutation=new_list_permut, list_neighsite=new_list_neighsite, list_neighisite=new_list_neighisite, structure=self.structure, valences=self.valences)
    else:
        new_list_ce_symbols = []
        new_list_csm = []
        new_list_permut = []
        new_list_neighsite = []
        new_list_neighisite = []
        for isite, _site in enumerate(self.structure):
            if isite in only_indices:
                new_list_ce_symbols.append(list_ce_symbols[isite])
                new_list_csm.append(list_csm[isite])
                new_list_permut.append(list_permut[isite])
                new_list_neighisite.append(self.list_neighisite[isite])
                new_list_neighsite.append(self.list_neighsite[isite])
            else:
                new_list_ce_symbols.append(None)
                new_list_csm.append(None)
                new_list_permut.append([])
                new_list_neighisite.append([])
                new_list_neighsite.append([])
        lse = LobsterLightStructureEnvironments.from_Lobster(list_ce_symbol=new_list_ce_symbols, list_csm=new_list_csm, list_permutation=new_list_permut, list_neighsite=new_list_neighsite, list_neighisite=new_list_neighisite, structure=self.structure, valences=self.valences)
    return lse