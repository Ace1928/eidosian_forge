from __future__ import annotations
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon
from monty.json import MontyDecoder, MSONable, jsanitize
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometries import AllCoordinationGeometries
from pymatgen.analysis.chemenv.coordination_environments.voronoi import DetailedVoronoiContainer
from pymatgen.analysis.chemenv.utils.chemenv_errors import ChemenvError
from pymatgen.analysis.chemenv.utils.defs_utils import AdditionalConditions
from pymatgen.core import Element, PeriodicNeighbor, PeriodicSite, Species, Structure
@classmethod
def from_structure_environments(cls, strategy, structure_environments, valences=None, valences_origin=None) -> Self:
    """
        Construct a LightStructureEnvironments object from a strategy and a StructureEnvironments object.

        Args:
            strategy: ChemEnv strategy used.
            structure_environments: StructureEnvironments object from which to construct the LightStructureEnvironments.
            valences: The valences of each site in the structure.
            valences_origin: How the valences were obtained (e.g. from the Bond-valence analysis or from the original
                structure).

        Returns:
            LightStructureEnvironments
        """
    structure = structure_environments.structure
    strategy.set_structure_environments(structure_environments=structure_environments)
    coordination_environments: list = [None] * len(structure)
    neighbors_sets: list = [None] * len(structure)
    _all_nbs_sites: list = []
    all_nbs_sites: list = []
    if valences is None:
        valences = structure_environments.valences
        if valences_origin is None:
            valences_origin = 'from_structure_environments'
    elif valences_origin is None:
        valences_origin = 'user-specified'
    for idx, site in enumerate(structure):
        site_ces_and_nbs_list = strategy.get_site_ce_fractions_and_neighbors(site, strategy_info=True)
        if site_ces_and_nbs_list is None:
            continue
        coordination_environments[idx] = []
        neighbors_sets[idx] = []
        site_ces = []
        site_nbs_sets: list = []
        for ce_and_neighbors in site_ces_and_nbs_list:
            _all_nbs_sites_indices = []
            ce_dict = {'ce_symbol': ce_and_neighbors['ce_symbol'], 'ce_fraction': ce_and_neighbors['ce_fraction']}
            if ce_and_neighbors['ce_dict'] is not None:
                csm = ce_and_neighbors['ce_dict']['other_symmetry_measures'][strategy.symmetry_measure_type]
            else:
                csm = None
            ce_dict['csm'] = csm
            ce_dict['permutation'] = (ce_and_neighbors.get('ce_dict') or {}).get('permutation')
            site_ces.append(ce_dict)
            neighbors = ce_and_neighbors['neighbors']
            for nb_site_and_index in neighbors:
                nb_site = nb_site_and_index['site']
                try:
                    n_all_nbs_sites_index = all_nbs_sites.index(nb_site)
                except ValueError:
                    nb_index_unitcell = nb_site_and_index['index']
                    diff = nb_site.frac_coords - structure[nb_index_unitcell].frac_coords
                    rounddiff = np.round(diff)
                    if not np.allclose(diff, rounddiff):
                        raise ValueError('Weird, differences between one site in a periodic image cell is not integer ...')
                    nb_image_cell = np.array(rounddiff, int)
                    n_all_nbs_sites_index = len(_all_nbs_sites)
                    _all_nbs_sites.append({'site': nb_site, 'index': nb_index_unitcell, 'image_cell': nb_image_cell})
                    all_nbs_sites.append(nb_site)
                _all_nbs_sites_indices.append(n_all_nbs_sites_index)
            nb_set = cls.NeighborsSet(structure=structure, isite=idx, all_nbs_sites=_all_nbs_sites, all_nbs_sites_indices=_all_nbs_sites_indices)
            site_nbs_sets.append(nb_set)
        coordination_environments[idx] = site_ces
        neighbors_sets[idx] = site_nbs_sets
    return cls(strategy=strategy, coordination_environments=coordination_environments, all_nbs_sites=_all_nbs_sites, neighbors_sets=neighbors_sets, structure=structure, valences=valences, valences_origin=valences_origin)