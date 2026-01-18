from __future__ import (absolute_import, division, print_function)
import os
import tempfile
import copy
from ansible_collections.dellemc.openmanage.plugins.module_utils.dellemc_idrac import iDRACConnection, idrac_auth_params
from ansible.module_utils.basic import AnsibleModule
def set_liason_share(idrac, module):
    idrac.use_redfish = True
    share_name = tempfile.gettempdir() + os.sep
    storage_share = file_share_manager.create_share_obj(share_path=share_name, isFolder=True)
    set_liason = idrac.config_mgr.set_liason_share(storage_share)
    if set_liason['Status'] == 'Failed':
        liason_data = set_liason.get('Data', set_liason)
        module.fail_json(msg=liason_data.get('Message', 'Failed to set Liason share'))