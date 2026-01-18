from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
import re
def add_to_valid_subset_list(valid_subset_list, subset_name, subset_options, fetch_all=False):
    if valid_subset_list is None:
        return []
    valid_subset = {}
    fields = query = limit = None
    detail = True
    count = -1
    if subset_options is not None:
        if 'fields' in subset_options and subset_options['fields'] is not None:
            temp = ''
            for item in subset_options['fields']:
                temp += item + ','
            fields = temp.strip(',')
        if 'detail' in subset_options and subset_options['detail'] is not None:
            detail = subset_options['detail']
        if 'limit' in subset_options:
            count = limit = subset_options['limit']
            if fetch_all is True:
                if subset_name in limit_not_supported:
                    limit = None
    if subset_options is not None and 'query' in subset_options:
        query = subset_options['query']
    valid_subset['name'] = subset_name.lower()
    valid_subset['fields'] = fields
    valid_subset['query'] = query
    valid_subset['limit'] = limit
    valid_subset['detail'] = detail
    valid_subset['count'] = count
    valid_subset_list.append(dict(valid_subset))
    return valid_subset_list