from __future__ import (absolute_import, division, print_function)
from collections import defaultdict
import platform
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts import timeout
def select_collector_classes(collector_names, all_fact_subsets):
    seen_collector_classes = set()
    selected_collector_classes = []
    for collector_name in collector_names:
        collector_classes = all_fact_subsets.get(collector_name, [])
        for collector_class in collector_classes:
            if collector_class not in seen_collector_classes:
                selected_collector_classes.append(collector_class)
                seen_collector_classes.add(collector_class)
    return selected_collector_classes