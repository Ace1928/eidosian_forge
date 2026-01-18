from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
import copy
def modify_limits(self, payload):
    """Modify IOPS and bandwidth limits of SDC's mapped to volume
            :param payload: Dict containing SDC ID's whose bandwidth and
                   IOPS is to modified
            :return: Boolean indicating if modifying limits is successful
        """
    try:
        changed = False
        if payload['bandwidth_limit'] is not None or payload['iops_limit'] is not None:
            self.powerflex_conn.volume.set_mapped_sdc_limits(**payload)
            changed = True
        return changed
    except Exception as e:
        errormsg = 'Modify bandwidth/iops limits of SDC %s operation failed with error %s' % (payload['sdc_id'], str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)