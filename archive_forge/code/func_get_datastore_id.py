from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible_collections.community.general.plugins.module_utils.opennebula import flatten, render
def get_datastore_id(module, client, requested_id, requested_name):
    datastore = get_datastore_by_id(module, client, requested_id) if requested_id else get_datastore_by_name(module, client, requested_name)
    if datastore:
        return datastore.ID
    else:
        return None