from __future__ import (absolute_import, division, print_function)
import fnmatch
import sys
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts import timeout
from ansible.module_utils.facts import collector
from ansible.module_utils.common.collections import is_string
def get_ansible_collector(all_collector_classes, namespace=None, filter_spec=None, gather_subset=None, gather_timeout=None, minimal_gather_subset=None):
    filter_spec = filter_spec or []
    gather_subset = gather_subset or ['all']
    gather_timeout = gather_timeout or timeout.DEFAULT_GATHER_TIMEOUT
    minimal_gather_subset = minimal_gather_subset or frozenset()
    collector_classes = collector.collector_classes_from_gather_subset(all_collector_classes=all_collector_classes, minimal_gather_subset=minimal_gather_subset, gather_subset=gather_subset, gather_timeout=gather_timeout)
    collectors = []
    for collector_class in collector_classes:
        collector_obj = collector_class(namespace=namespace)
        collectors.append(collector_obj)
    collector_meta_data_collector = CollectorMetaDataCollector(gather_subset=gather_subset, module_setup=True)
    collectors.append(collector_meta_data_collector)
    fact_collector = AnsibleFactCollector(collectors=collectors, filter_spec=filter_spec, namespace=namespace)
    return fact_collector