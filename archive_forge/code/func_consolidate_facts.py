from __future__ import absolute_import, division, print_function
import itertools
from ansible.errors import AnsibleFilterError
def consolidate_facts(data_sources, all_values):
    """Iterate over all the data sources and consolidate the data

    Args:
        data_sources (list): supplied data sources
        all_values (set): a set of keys to iterate over

    Returns:
        list: list of consolidated data
    """
    consolidated_facts = {}
    for data_source in data_sources:
        match_key = data_source['match_key']
        source = data_source['name']
        data_dict = {d[match_key]: d for d in data_source['data'] if match_key in d}
        for value in sorted(all_values):
            if value not in consolidated_facts:
                consolidated_facts[value] = {}
            consolidated_facts[value][source] = data_dict.get(value, {})
    return consolidated_facts