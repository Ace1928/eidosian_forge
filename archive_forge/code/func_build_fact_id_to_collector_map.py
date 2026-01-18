from __future__ import (absolute_import, division, print_function)
from collections import defaultdict
import platform
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts import timeout
def build_fact_id_to_collector_map(collectors_for_platform):
    fact_id_to_collector_map = defaultdict(list)
    aliases_map = defaultdict(set)
    for collector_class in collectors_for_platform:
        primary_name = collector_class.name
        fact_id_to_collector_map[primary_name].append(collector_class)
        for fact_id in collector_class._fact_ids:
            fact_id_to_collector_map[fact_id].append(collector_class)
            aliases_map[primary_name].add(fact_id)
    return (fact_id_to_collector_map, aliases_map)