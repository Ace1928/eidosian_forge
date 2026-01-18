from __future__ import (absolute_import, division, print_function)
import os
import tempfile
import json
from ansible_collections.dellemc.openmanage.plugins.module_utils.dellemc_idrac import iDRACConnection, idrac_auth_params
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def run_setup_idrac_csior(idrac, module):
    """
    Get Lifecycle Controller status

    Keyword arguments:
    idrac  -- iDRAC handle
    module -- Ansible module
    """
    idrac.use_redfish = True
    share_path = tempfile.gettempdir() + os.sep
    upd_share = file_share_manager.create_share_obj(share_path=share_path, isFolder=True)
    if not upd_share.IsValid:
        module.fail_json(msg='Unable to access the share. Ensure that the share name, share mount, and share credentials provided are correct.')
    set_liason = idrac.config_mgr.set_liason_share(upd_share)
    if set_liason['Status'] == 'Failed':
        try:
            message = set_liason['Data']['Message']
        except (IndexError, KeyError):
            message = set_liason['Message']
        module.fail_json(msg=message)
    if module.params['csior'] == 'Enabled':
        idrac.config_mgr.enable_csior()
    elif module.params['csior'] == 'Disabled':
        idrac.config_mgr.disable_csior()
    if module.check_mode:
        status = idrac.config_mgr.is_change_applicable()
        if status.get('changes_applicable'):
            module.exit_json(msg='Changes found to commit!', changed=True)
        else:
            module.exit_json(msg='No changes found to commit!')
    else:
        return idrac.config_mgr.apply_changes(reboot=False)