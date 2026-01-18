from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
import copy
def is_id_new_name_in_add_mdm(self):
    """ Check whether mdm_id or mdm_new_name present in Add standby MDM"""
    if self.module.params['mdm_id'] or self.module.params['mdm_new_name']:
        err_msg = 'Parameters mdm_id/mdm_new_name are not allowed while adding a standby MDM. Please try with valid parameters to add a standby MDM.'
        LOG.error(err_msg)
        self.module.fail_json(msg=err_msg)