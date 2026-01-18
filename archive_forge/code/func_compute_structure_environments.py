from __future__ import annotations
import itertools
import logging
import time
from random import shuffle
from typing import TYPE_CHECKING
import numpy as np
from numpy.linalg import norm, svd
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import MultiWeightsChemenvStrategy
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometries import (
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import (
from pymatgen.analysis.chemenv.coordination_environments.voronoi import DetailedVoronoiContainer
from pymatgen.analysis.chemenv.utils.coordination_geometry_utils import (
from pymatgen.core import Lattice, Species, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.due import Doi, due
def compute_structure_environments(self, excluded_atoms=None, only_atoms=None, only_cations=True, only_indices=None, maximum_distance_factor=PRESETS['DEFAULT']['maximum_distance_factor'], minimum_angle_factor=PRESETS['DEFAULT']['minimum_angle_factor'], max_cn=None, min_cn=None, only_symbols=None, valences='undefined', additional_conditions=None, info=None, timelimit=None, initial_structure_environments=None, get_from_hints=False, voronoi_normalized_distance_tolerance=PRESETS['DEFAULT']['voronoi_normalized_distance_tolerance'], voronoi_normalized_angle_tolerance=PRESETS['DEFAULT']['voronoi_normalized_angle_tolerance'], voronoi_distance_cutoff=None, recompute=None, optimization=PRESETS['DEFAULT']['optimization']):
    """
        Computes and returns the StructureEnvironments object containing all the information about the coordination
        environments in the structure

        Args:
            excluded_atoms: Atoms for which the coordination geometries does not have to be identified
            only_atoms: If not set to None, atoms for which the coordination geometries have to be identified
            only_cations: If set to True, will only compute environments for cations
            only_indices: If not set to None, will only compute environments the atoms of the given indices
            maximum_distance_factor: If not set to None, neighbors beyond
                maximum_distance_factor*closest_neighbor_distance are not considered
            minimum_angle_factor: If not set to None, neighbors for which the angle is lower than
                minimum_angle_factor*largest_angle_neighbor are not considered
            max_cn: maximum coordination number to be considered
            min_cn: minimum coordination number to be considered
            only_symbols: if not set to None, consider only coordination environments with the given symbols
            valences: valences of the atoms
            additional_conditions: additional conditions to be considered in the bonds (example : only bonds
                between cation and anion
            info: additional info about the calculation
            timelimit: time limit (in secs) after which the calculation of the StructureEnvironments object stops
            initial_structure_environments: initial StructureEnvironments object (most probably incomplete)
            get_from_hints: whether to add neighbors sets from "hints" (e.g. capped environment => test the
                neighbors without the cap)
            voronoi_normalized_distance_tolerance: tolerance for the normalized distance used to distinguish
                neighbors sets
            voronoi_normalized_angle_tolerance: tolerance for the normalized angle used to distinguish
                neighbors sets
            voronoi_distance_cutoff: determines distance of considered neighbors. Especially important to increase it
                for molecules in a box.
            recompute: whether to recompute the sites already computed (when initial_structure_environments
                is not None)
            optimization: optimization algorithm

        Returns:
            The StructureEnvironments object containing all the information about the coordination
            environments in the structure.
        """
    time_init = time.process_time()
    if info is None:
        info = {}
    info.update(local_geometry_finder={'parameters': {'centering_type': self.centering_type, 'include_central_site_in_centroid': self.include_central_site_in_centroid, 'structure_refinement': self.structure_refinement, 'spg_analyzer_options': self.spg_analyzer_options}})
    if only_symbols is not None:
        self.allcg = AllCoordinationGeometries(permutations_safe_override=self.permutations_safe_override, only_symbols=only_symbols)
    if valences == 'undefined':
        first_site = self.structure[0]
        try:
            sp = first_site.specie
            if isinstance(sp, Species):
                self.valences = [int(site.specie.oxi_state) for site in self.structure]
            else:
                self.valences = valences
        except AttributeError:
            self.valences = valences
    else:
        self.valences = valences
    self.equivalent_sites = [[site] for site in self.structure]
    self.struct_sites_to_irreducible_site_list_map = list(range(len(self.structure)))
    self.sites_map = list(range(len(self.structure)))
    indices = list(range(len(self.structure)))
    if only_cations and self.valences != 'undefined':
        sites_indices = [idx for idx in indices if self.valences[idx] >= 0]
    else:
        sites_indices = list(indices)
    if only_atoms is not None:
        sites_indices = [idx for idx in sites_indices if any((at in [sp.symbol for sp in self.structure[idx].species] for at in only_atoms))]
    if excluded_atoms:
        sites_indices = [idx for idx in sites_indices if not any((at in [sp.symbol for sp in self.structure[idx].species] for at in excluded_atoms))]
    if only_indices is not None:
        sites_indices = [isite for isite in indices if isite in only_indices]
    logging.debug('Getting DetailedVoronoiContainer')
    if voronoi_normalized_distance_tolerance is None:
        normalized_distance_tolerance = DetailedVoronoiContainer.default_normalized_distance_tolerance
    else:
        normalized_distance_tolerance = voronoi_normalized_distance_tolerance
    if voronoi_normalized_angle_tolerance is None:
        normalized_angle_tolerance = DetailedVoronoiContainer.default_normalized_angle_tolerance
    else:
        normalized_angle_tolerance = voronoi_normalized_angle_tolerance
    if voronoi_distance_cutoff is None:
        voronoi_distance_cutoff = DetailedVoronoiContainer.default_voronoi_cutoff
    self.detailed_voronoi = DetailedVoronoiContainer(self.structure, isites=sites_indices, valences=self.valences, maximum_distance_factor=maximum_distance_factor, minimum_angle_factor=minimum_angle_factor, additional_conditions=additional_conditions, normalized_distance_tolerance=normalized_distance_tolerance, normalized_angle_tolerance=normalized_angle_tolerance, voronoi_cutoff=voronoi_distance_cutoff)
    logging.debug('DetailedVoronoiContainer has been set up')
    if initial_structure_environments is not None:
        struct_envs = initial_structure_environments
        if struct_envs.structure != self.structure:
            raise ValueError('Structure is not the same in initial_structure_environments')
        if struct_envs.voronoi != self.detailed_voronoi:
            if self.detailed_voronoi.is_close_to(struct_envs.voronoi):
                self.detailed_voronoi = struct_envs.voronoi
            else:
                raise ValueError('Detailed Voronoi is not the same in initial_structure_environments')
        struct_envs.info = info
    else:
        struct_envs = StructureEnvironments(voronoi=self.detailed_voronoi, valences=self.valences, sites_map=self.sites_map, equivalent_sites=self.equivalent_sites, ce_list=[None] * len(self.structure), structure=self.structure, info=info)
    if min_cn is None:
        min_cn = 1
    if max_cn is None:
        max_cn = 20
    all_cns = range(min_cn, max_cn + 1)
    do_recompute = False
    if recompute is not None:
        if 'cns' in recompute:
            cns_to_recompute = recompute['cns']
            all_cns = list(set(all_cns).intersection(cns_to_recompute))
        do_recompute = True
    max_time_one_site = 0.0
    break_it = False
    if optimization > 0:
        self.detailed_voronoi.local_planes = [None] * len(self.structure)
        self.detailed_voronoi.separations = [None] * len(self.structure)
    for isite, site in enumerate(self.structure):
        if isite not in sites_indices:
            logging.debug(f' ... in site #{isite}/{len(self.structure)} ({site.species_string}) : skipped')
            continue
        if break_it:
            logging.debug(f' ... in site #{isite}/{len(self.structure)} ({site.species_string}) : skipped (timelimit)')
            continue
        logging.debug(f' ... in site #{isite}/{len(self.structure)} ({site.species_string})')
        t1 = time.process_time()
        if optimization > 0:
            self.detailed_voronoi.local_planes[isite] = {}
            self.detailed_voronoi.separations[isite] = {}
        struct_envs.init_neighbors_sets(isite=isite, additional_conditions=additional_conditions, valences=valences)
        to_add_from_hints = []
        nb_sets_info = {}
        for cn, nb_sets in struct_envs.neighbors_sets[isite].items():
            if cn not in all_cns:
                continue
            for inb_set, nb_set in enumerate(nb_sets):
                logging.debug(f'    ... getting environments for nb_set ({cn}, {inb_set})')
                t_nbset1 = time.process_time()
                ce = self.update_nb_set_environments(se=struct_envs, isite=isite, cn=cn, inb_set=inb_set, nb_set=nb_set, recompute=do_recompute, optimization=optimization)
                t_nbset2 = time.process_time()
                nb_sets_info.setdefault(cn, {})
                nb_sets_info[cn][inb_set] = {'time': t_nbset2 - t_nbset1}
                if get_from_hints:
                    for cg_symbol, cg_dict in ce:
                        cg = self.allcg[cg_symbol]
                        if cg.neighbors_sets_hints is None:
                            continue
                        logging.debug(f'       ... getting hints from cg with mp_symbol {cg_symbol!r} ...')
                        hints_info = {'csm': cg_dict['symmetry_measure'], 'nb_set': nb_set, 'permutation': cg_dict['permutation']}
                        for nb_sets_hints in cg.neighbors_sets_hints:
                            suggested_nb_set_voronoi_indices = nb_sets_hints.hints(hints_info)
                            for idx_new, new_nb_set_voronoi_indices in enumerate(suggested_nb_set_voronoi_indices):
                                logging.debug(f'           hint # {idx_new}')
                                new_nb_set = struct_envs.NeighborsSet(structure=struct_envs.structure, isite=isite, detailed_voronoi=struct_envs.voronoi, site_voronoi_indices=new_nb_set_voronoi_indices, sources={'origin': 'nb_set_hints', 'hints_type': nb_sets_hints.hints_type, 'suggestion_index': idx_new, 'cn_map_source': [cn, inb_set], 'cg_source_symbol': cg_symbol})
                                cn_new_nb_set = len(new_nb_set)
                                if max_cn is not None and cn_new_nb_set > max_cn:
                                    continue
                                if min_cn is not None and cn_new_nb_set < min_cn:
                                    continue
                                if new_nb_set in [ta['new_nb_set'] for ta in to_add_from_hints]:
                                    has_nb_set = True
                                elif cn_new_nb_set not in struct_envs.neighbors_sets[isite]:
                                    has_nb_set = False
                                else:
                                    has_nb_set = new_nb_set in struct_envs.neighbors_sets[isite][cn_new_nb_set]
                                if not has_nb_set:
                                    to_add_from_hints.append({'isite': isite, 'new_nb_set': new_nb_set, 'cn_new_nb_set': cn_new_nb_set})
                                    logging.debug('              => to be computed')
                                else:
                                    logging.debug('              => already present')
        logging.debug('    ... getting environments for nb_sets added from hints')
        for missing_nb_set_to_add in to_add_from_hints:
            struct_envs.add_neighbors_set(isite=isite, nb_set=missing_nb_set_to_add['new_nb_set'])
        for missing_nb_set_to_add in to_add_from_hints:
            isite_new_nb_set = missing_nb_set_to_add['isite']
            cn_new_nb_set = missing_nb_set_to_add['cn_new_nb_set']
            new_nb_set = missing_nb_set_to_add['new_nb_set']
            inew_nb_set = struct_envs.neighbors_sets[isite_new_nb_set][cn_new_nb_set].index(new_nb_set)
            logging.debug(f'    ... getting environments for nb_set ({cn_new_nb_set}, {inew_nb_set}) - from hints')
            t_nbset1 = time.process_time()
            self.update_nb_set_environments(se=struct_envs, isite=isite_new_nb_set, cn=cn_new_nb_set, inb_set=inew_nb_set, nb_set=new_nb_set, optimization=optimization)
            t_nbset2 = time.process_time()
            if cn not in nb_sets_info:
                nb_sets_info[cn] = {}
            nb_sets_info[cn][inew_nb_set] = {'time': t_nbset2 - t_nbset1}
        t2 = time.process_time()
        struct_envs.update_site_info(isite=isite, info_dict={'time': t2 - t1, 'nb_sets_info': nb_sets_info})
        if timelimit is not None:
            time_elapsed = t2 - time_init
            time_left = timelimit - time_elapsed
            if time_left < 2.0 * max_time_one_site:
                break_it = True
        max_time_one_site = max(max_time_one_site, t2 - t1)
        logging.debug(f'    ... computed in {t2 - t1:.2f} seconds')
    time_end = time.process_time()
    logging.debug(f'    ... compute_structure_environments ended in {time_end - time_init:.2f} seconds')
    return struct_envs