from __future__ import (absolute_import, division, print_function)
from collections import defaultdict
import platform
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts import timeout
def collector_classes_from_gather_subset(all_collector_classes=None, valid_subsets=None, minimal_gather_subset=None, gather_subset=None, gather_timeout=None, platform_info=None):
    """return a list of collector classes that match the args"""
    all_collector_classes = all_collector_classes or []
    minimal_gather_subset = minimal_gather_subset or frozenset()
    platform_info = platform_info or {'system': platform.system()}
    gather_timeout = gather_timeout or timeout.DEFAULT_GATHER_TIMEOUT
    timeout.GATHER_TIMEOUT = gather_timeout
    valid_subsets = valid_subsets or frozenset()
    aliases_map = defaultdict(set)
    compat_platforms = [platform_info, {'system': 'Generic'}]
    collectors_for_platform = find_collectors_for_platform(all_collector_classes, compat_platforms)
    all_fact_subsets, aliases_map = build_fact_id_to_collector_map(collectors_for_platform)
    all_valid_subsets = frozenset(all_fact_subsets.keys())
    collector_names = get_collector_names(valid_subsets=all_valid_subsets, minimal_gather_subset=minimal_gather_subset, gather_subset=gather_subset, aliases_map=aliases_map, platform_info=platform_info)
    complete_collector_names = _solve_deps(collector_names, all_fact_subsets)
    dep_map = build_dep_data(complete_collector_names, all_fact_subsets)
    ordered_deps = tsort(dep_map)
    ordered_collector_names = [x[0] for x in ordered_deps]
    selected_collector_classes = select_collector_classes(ordered_collector_names, all_fact_subsets)
    return selected_collector_classes