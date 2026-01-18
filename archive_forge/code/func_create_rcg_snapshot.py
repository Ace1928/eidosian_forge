from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
def create_rcg_snapshot(self, rcg_id):
    """Create RCG snapshot
            :param rcg_id: Unique identifier of the RCG.
            :return: Boolean indicating if create snapshot operation is successful
        """
    try:
        if not self.module.check_mode:
            self.powerflex_conn.replication_consistency_group.create_snapshot(rcg_id=rcg_id)
        return True
    except Exception as e:
        errormsg = 'Create RCG snapshot for RCG with id {0} operation failed with error {1}'.format(rcg_id, str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)