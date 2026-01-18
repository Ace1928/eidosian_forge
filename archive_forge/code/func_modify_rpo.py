from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
def modify_rpo(self, rcg_id, rpo):
    """Modify rpo
            :param rcg_id: Unique identifier of the RCG.
            :param rpo: rpo value in seconds
            :return: Boolean indicates if modify rpo is successful
        """
    try:
        if not self.module.check_mode:
            self.powerflex_conn.replication_consistency_group.modify_rpo(rcg_id, rpo)
        return True
    except Exception as e:
        errormsg = 'Modify rpo for replication consistency group {0} failed with error {1}'.format(rcg_id, str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)