from __future__ import (absolute_import, division, print_function)
import os
from ansible_collections.dellemc.openmanage.plugins.module_utils.dellemc_idrac import iDRACConnection, idrac_auth_params
from ansible.module_utils.basic import AnsibleModule
def run_boot_to_network_iso(idrac, module):
    """Boot to a network ISO image"""
    try:
        share_name = module.params['share_name']
        if share_name is None:
            share_name = ''
        share_obj = FileOnShare(remote='{0}{1}{2}'.format(share_name, os.sep, module.params['iso_image']), isFolder=False, creds=UserCredentials(module.params['share_user'], module.params['share_password']))
        cim_exp_duration = minutes_to_cim_format(module, module.params['expose_duration'])
        boot_status = idrac.config_mgr.boot_to_network_iso(share_obj, '', expose_duration=cim_exp_duration)
        if not boot_status.get('Status', False) == 'Success':
            module.fail_json(msg=boot_status)
    except Exception as e:
        module.fail_json(msg=str(e))
    return boot_status