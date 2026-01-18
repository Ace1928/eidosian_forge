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
@classmethod
def from_Lobster(cls, list_ce_symbol, list_csm, list_permutation, list_neighsite, list_neighisite, structure: Structure, valences=None) -> Self:
    """
        Will set up a LightStructureEnvironments from Lobster.

        Args:
            structure: Structure object
            list_ce_symbol: list of symbols for coordination environments
            list_csm: list of continuous symmetry measures
            list_permutation: list of permutations
            list_neighsite: list of neighboring sites
            list_neighisite: list of neighboring isites (number of a site)
            valences: list of valences

        Returns:
            LobsterLightStructureEnvironments
        """
    strategy = None
    valences_origin = 'user-defined'
    coordination_environments = []
    all_nbs_sites = []
    all_nbs_sites_indices = []
    neighbors_sets = []
    counter = 0
    for isite in range(len(structure)):
        all_nbs_sites_indices_here = []
        if list_ce_symbol is not None:
            ce_dict = {'ce_symbol': list_ce_symbol[isite], 'ce_fraction': 1.0, 'csm': list_csm[isite], 'permutation': list_permutation[isite]}
        else:
            ce_dict = None
        if list_neighisite[isite] is not None:
            for ineighsite, neighsite in enumerate(list_neighsite[isite]):
                diff = neighsite.frac_coords - structure[list_neighisite[isite][ineighsite]].frac_coords
                rounddiff = np.round(diff)
                if not np.allclose(diff, rounddiff):
                    raise ValueError('Weird, differences between one site in a periodic image cell is not integer ...')
                nb_image_cell = np.array(rounddiff, int)
                all_nbs_sites_indices_here.append(counter)
                all_nbs_sites.append({'site': neighsite, 'index': list_neighisite[isite][ineighsite], 'image_cell': nb_image_cell})
                counter = counter + 1
            all_nbs_sites_indices.append(all_nbs_sites_indices_here)
        else:
            all_nbs_sites.append({'site': None, 'index': None, 'image_cell': None})
            all_nbs_sites_indices.append([])
        if list_neighisite[isite] is not None:
            nb_set = cls.NeighborsSet(structure=structure, isite=isite, all_nbs_sites=all_nbs_sites, all_nbs_sites_indices=all_nbs_sites_indices[isite])
        else:
            nb_set = cls.NeighborsSet(structure=structure, isite=isite, all_nbs_sites=[], all_nbs_sites_indices=[])
        coordination_environments.append([ce_dict])
        neighbors_sets.append([nb_set])
    return cls(strategy=strategy, coordination_environments=coordination_environments, all_nbs_sites=all_nbs_sites, neighbors_sets=neighbors_sets, structure=structure, valences=valences, valences_origin=valences_origin)