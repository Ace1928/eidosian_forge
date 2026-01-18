from __future__ import absolute_import, division, print_function
from ansible.errors import AnsibleFilterError
def pop_ace(data, filter_options, match_criteria):
    if not isinstance(data, (list, dict)):
        _raise_error('Input is not valid for pop_ace')
    cleared_data, removed_data = _pop_ace(data, filter_options, match_criteria)
    data = {'clean_acls': cleared_data, 'removed_aces': removed_data}
    return data