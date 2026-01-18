from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
import copy
def modify_access_mode(self, vol_id, access_mode_list):
    """Modify access mode of SDCs mapped to volume
            :param vol_id: The volume id
            :param access_mode_list: List containing SDC ID's
             whose access mode is to modified
            :return: Boolean indicating if modifying access
             mode is successful
        """
    try:
        changed = False
        for temp in access_mode_list:
            if temp['accessMode']:
                self.powerflex_conn.volume.set_access_mode_for_sdc(volume_id=vol_id, sdc_id=temp['sdc_id'], access_mode=temp['accessMode'])
                changed = True
        return changed
    except Exception as e:
        errormsg = 'Modify access mode of SDC operation failed with error {0}'.format(str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)