from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
import re
def handle_all_subset(info_subset, valid_subset_list, subset_options):
    if valid_subset_list is None or info_subset is None:
        return []
    msg = "Subset options 'fields and query' cannot be used with 'all' subset. Only 'limit and detail' option can be used."
    if subset_options is not None:
        if 'fields' in subset_options or 'query' in subset_options:
            raise Exception(msg)
    for key, value in info_subset.items():
        if is_subset_already_added(key, valid_subset_list) is False and (key != 'minimum' and key != 'config' and (key != 'snapshots')):
            add_to_valid_subset_list(valid_subset_list, key, subset_options, True)
    return valid_subset_list