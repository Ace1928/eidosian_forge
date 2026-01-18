from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def get_arrays(array):
    """Get Connected Arrays"""
    arrays = []
    array_details = array.list_array_connections()
    api_version = array._list_available_rest_versions()
    for arraycnt in range(0, len(array_details)):
        if P53_API_VERSION in api_version:
            if array_details[arraycnt]['status'] in ['connected', 'partially_connected']:
                arrays.append(array_details[arraycnt]['array_name'])
        elif array_details[arraycnt]['connected']:
            arrays.append(array_details[arraycnt]['array_name'])
    return arrays