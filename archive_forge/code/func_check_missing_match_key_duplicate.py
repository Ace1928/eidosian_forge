from __future__ import absolute_import, division, print_function
import itertools
from ansible.errors import AnsibleFilterError
@fail_on_filter
def check_missing_match_key_duplicate(data_sources, fail_missing_match_key, fail_duplicate):
    """Check if the match_key specified is present in all the supplied data,
    also check for duplicate data accross all the data sources

    Args:
        data_sources (list): list of dicts as data sources
        fail_missing_match_key (bool): Fails if match_keys not present in data set
        fail_duplicate (bool): Fails if duplicate data present in a data
    Returns:
        list: list of unique keys based on specified match_keys
    """
    results, errors_match_key, errors_duplicate = ([], [], [])
    for ds_idx, data_source in enumerate(data_sources, start=1):
        match_key = data_source['match_key']
        ds_values = []
        for dd_idx, data_dict in enumerate(data_source['data'], start=1):
            try:
                ds_values.append(data_dict[match_key])
            except KeyError:
                if fail_missing_match_key:
                    errors_match_key.append("missing match key '{match_key}' in data source {ds_idx} in list entry {dd_idx}".format(match_key=match_key, ds_idx=ds_idx, dd_idx=dd_idx))
                continue
        if sorted(set(ds_values)) != sorted(ds_values) and fail_duplicate:
            errors_duplicate.append('duplicate values in data source {ds_idx}'.format(ds_idx=ds_idx))
        results.append(set(ds_values))
    return (results, {'fail_missing_match_key': errors_match_key, 'fail_duplicate': errors_duplicate})