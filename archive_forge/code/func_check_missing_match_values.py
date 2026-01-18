from __future__ import absolute_import, division, print_function
import itertools
from ansible.errors import AnsibleFilterError
@fail_on_filter
def check_missing_match_values(matched_keys, fail_missing_match_value):
    """Checks values to match be consistent over all the whole data source

    Args:
        matched_keys (list): list of unique keys based on specified match_keys
        fail_missing_match_value (bool): Fail if match_key value is missing in a data set
    Returns:
        set: set of unique values
    """
    all_values = set(itertools.chain.from_iterable(matched_keys))
    if not fail_missing_match_value:
        return (all_values, {})
    errors_match_values = []
    for ds_idx, ds_values in enumerate(matched_keys, start=1):
        missing_match = all_values - ds_values
        if missing_match:
            m_matches = ', '.join(missing_match)
            errors_match_values.append('missing match value {m_matches} in data source {ds_idx}'.format(ds_idx=ds_idx, m_matches=m_matches))
    return (all_values, {'fail_missing_match_value': errors_match_values})