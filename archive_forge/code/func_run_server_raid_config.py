from __future__ import (absolute_import, division, print_function)
import os
import tempfile
import copy
from ansible_collections.dellemc.openmanage.plugins.module_utils.dellemc_idrac import iDRACConnection, idrac_auth_params
from ansible.module_utils.basic import AnsibleModule
def run_server_raid_config(idrac, module):
    if module.params['state'] == 'view':
        storage_status = view_storage(idrac, module)
    if module.params['state'] == 'create':
        set_liason_share(idrac, module)
        storage_status = create_storage(idrac, module)
    if module.params['state'] == 'delete':
        set_liason_share(idrac, module)
        storage_status = delete_storage(idrac, module)
    return storage_status