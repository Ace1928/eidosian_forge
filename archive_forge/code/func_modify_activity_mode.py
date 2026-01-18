from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
def modify_activity_mode(self, rcg_id, rcg_details, activity_mode):
    """Modify activity mode
            :param rcg_id: Unique identifier of the RCG.
            :param rcg_details: RCG details.
            :param activity_mode: RCG activity mode.
            :return: Boolean indicates if modify operation is successful
        """
    try:
        if activity_mode == 'Active' and rcg_details['localActivityState'].lower() == 'inactive':
            if not self.module.check_mode:
                self.powerflex_conn.replication_consistency_group.activate(rcg_id)
            return True
        elif activity_mode == 'Inactive' and rcg_details['localActivityState'].lower() == 'active':
            if not self.module.check_mode:
                rcg_details = self.powerflex_conn.replication_consistency_group.inactivate(rcg_id)
            return True
    except Exception as e:
        errormsg = 'Modify activity_mode for replication consistency group {0} failed with error {1}'.format(rcg_id, str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)