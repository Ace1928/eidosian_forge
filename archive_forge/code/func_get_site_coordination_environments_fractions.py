from __future__ import annotations
import abc
import os
from typing import TYPE_CHECKING, ClassVar
import numpy as np
from monty.json import MSONable
from scipy.stats import gmean
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometries import AllCoordinationGeometries
from pymatgen.analysis.chemenv.coordination_environments.voronoi import DetailedVoronoiContainer
from pymatgen.analysis.chemenv.utils.chemenv_errors import EquivalentSiteSearchError
from pymatgen.analysis.chemenv.utils.coordination_geometry_utils import get_lower_and_upper_f
from pymatgen.analysis.chemenv.utils.defs_utils import AdditionalConditions
from pymatgen.analysis.chemenv.utils.func_utils import (
from pymatgen.core.operations import SymmOp
from pymatgen.core.sites import PeriodicSite
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def get_site_coordination_environments_fractions(self, site, isite=None, dequivsite=None, dthissite=None, mysym=None, ordered=True, min_fraction=0, return_maps=True, return_strategy_dict_info=False, return_all=False):
    """Get the coordination environments of a given site and additional information.

        Args:
            site: Site for which coordination environment is needed.
            isite: Index of the site.
            dequivsite: Translation of the equivalent site.
            dthissite: Translation of this site.
            mysym: Symmetry to be applied.
            ordered: Whether to order the list by fractions.
            min_fraction: Minimum fraction to include in the list
            return_maps: Whether to return cn_maps (identifies all the NeighborsSet used).
            return_strategy_dict_info: Whether to add the info about the strategy used.

        Returns:
            List of Dict with coordination environment, fraction and additional info.
        """
    if isite is None or dequivsite is None or dthissite is None or (mysym is None):
        isite, dequivsite, dthissite, mysym = self.equivalent_site_index_and_transform(site)
    site_nb_sets = self.structure_environments.neighbors_sets[isite]
    if site_nb_sets is None:
        return None
    cn_maps = []
    for cn, nb_sets in site_nb_sets.items():
        for inb_set in range(len(nb_sets)):
            cn_maps.append((cn, inb_set))
    weights_additional_info = {'weights': {isite: {}}}
    for wdict in self.ordered_weights:
        cn_maps_new = []
        weight = wdict['weight']
        weight_name = wdict['name']
        for cn_map in cn_maps:
            nb_set = site_nb_sets[cn_map[0]][cn_map[1]]
            w_nb_set = weight.weight(nb_set=nb_set, structure_environments=self.structure_environments, cn_map=cn_map, additional_info=weights_additional_info)
            if cn_map not in weights_additional_info['weights'][isite]:
                weights_additional_info['weights'][isite][cn_map] = {}
            weights_additional_info['weights'][isite][cn_map][weight_name] = w_nb_set
            if return_all or w_nb_set > 0:
                cn_maps_new.append(cn_map)
        cn_maps = cn_maps_new
    for cn_map, weights in weights_additional_info['weights'][isite].items():
        weights_additional_info['weights'][isite][cn_map]['Product'] = np.prod(list(weights.values()))
    w_nb_sets = {cn_map: weights['Product'] for cn_map, weights in weights_additional_info['weights'][isite].items()}
    w_nb_sets_total = np.sum(list(w_nb_sets.values()))
    nb_sets_fractions = {cn_map: w_nb_set / w_nb_sets_total for cn_map, w_nb_set in w_nb_sets.items()}
    for cn_map in weights_additional_info['weights'][isite]:
        weights_additional_info['weights'][isite][cn_map]['NbSetFraction'] = nb_sets_fractions[cn_map]
    ce_symbols = []
    ce_dicts = []
    ce_fractions = []
    ce_dict_fractions = []
    ce_maps = []
    site_ce_list = self.structure_environments.ce_list[isite]
    if return_all:
        for cn_map, nb_set_fraction in nb_sets_fractions.items():
            cn = cn_map[0]
            inb_set = cn_map[1]
            site_ce_nb_set = site_ce_list[cn][inb_set]
            if site_ce_nb_set is None:
                continue
            mingeoms = site_ce_nb_set.minimum_geometries(symmetry_measure_type=self.symmetry_measure_type)
            if len(mingeoms) > 0:
                csms = [ce_dict['other_symmetry_measures'][self.symmetry_measure_type] for ce_symbol, ce_dict in mingeoms]
                fractions = self.ce_estimator_fractions(csms)
                if fractions is None:
                    ce_symbols.append(f'UNCLEAR:{cn}')
                    ce_dicts.append(None)
                    ce_fractions.append(nb_set_fraction)
                    all_weights = weights_additional_info['weights'][isite][cn_map]
                    dict_fractions = dict(all_weights.items())
                    dict_fractions['CEFraction'] = None
                    dict_fractions['Fraction'] = nb_set_fraction
                    ce_dict_fractions.append(dict_fractions)
                    ce_maps.append(cn_map)
                else:
                    for ifraction, fraction in enumerate(fractions):
                        ce_symbols.append(mingeoms[ifraction][0])
                        ce_dicts.append(mingeoms[ifraction][1])
                        ce_fractions.append(nb_set_fraction * fraction)
                        all_weights = weights_additional_info['weights'][isite][cn_map]
                        dict_fractions = dict(all_weights.items())
                        dict_fractions['CEFraction'] = fraction
                        dict_fractions['Fraction'] = nb_set_fraction * fraction
                        ce_dict_fractions.append(dict_fractions)
                        ce_maps.append(cn_map)
            else:
                ce_symbols.append(f'UNCLEAR:{cn}')
                ce_dicts.append(None)
                ce_fractions.append(nb_set_fraction)
                all_weights = weights_additional_info['weights'][isite][cn_map]
                dict_fractions = dict(all_weights.items())
                dict_fractions['CEFraction'] = None
                dict_fractions['Fraction'] = nb_set_fraction
                ce_dict_fractions.append(dict_fractions)
                ce_maps.append(cn_map)
    else:
        for cn_map, nb_set_fraction in nb_sets_fractions.items():
            if nb_set_fraction > 0:
                cn = cn_map[0]
                inb_set = cn_map[1]
                site_ce_nb_set = site_ce_list[cn][inb_set]
                mingeoms = site_ce_nb_set.minimum_geometries(symmetry_measure_type=self._symmetry_measure_type)
                csms = [ce_dict['other_symmetry_measures'][self._symmetry_measure_type] for ce_symbol, ce_dict in mingeoms]
                fractions = self.ce_estimator_fractions(csms)
                for ifraction, fraction in enumerate(fractions):
                    if fraction > 0:
                        ce_symbols.append(mingeoms[ifraction][0])
                        ce_dicts.append(mingeoms[ifraction][1])
                        ce_fractions.append(nb_set_fraction * fraction)
                        all_weights = weights_additional_info['weights'][isite][cn_map]
                        dict_fractions = dict(all_weights.items())
                        dict_fractions['CEFraction'] = fraction
                        dict_fractions['Fraction'] = nb_set_fraction * fraction
                        ce_dict_fractions.append(dict_fractions)
                        ce_maps.append(cn_map)
    indices = np.argsort(ce_fractions)[::-1] if ordered else list(range(len(ce_fractions)))
    fractions_info_list = [{'ce_symbol': ce_symbols[ii], 'ce_dict': ce_dicts[ii], 'ce_fraction': ce_fractions[ii]} for ii in indices if ce_fractions[ii] >= min_fraction]
    if return_maps:
        for ifinfo, ii in enumerate(indices):
            if ce_fractions[ii] >= min_fraction:
                fractions_info_list[ifinfo]['ce_map'] = ce_maps[ii]
    if return_strategy_dict_info:
        for ifinfo, ii in enumerate(indices):
            if ce_fractions[ii] >= min_fraction:
                fractions_info_list[ifinfo]['strategy_info'] = ce_dict_fractions[ii]
    return fractions_info_list