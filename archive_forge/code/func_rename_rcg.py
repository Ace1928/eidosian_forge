from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
def rename_rcg(self, rcg_id, rcg_details, new_name):
    """Rename rcg
            :param rcg_id: Unique identifier of the RCG.
            :param rcg_details: RCG details
            :param new_name: RCG name to rename to.
            :return: Boolean indicates if rename is successful
        """
    try:
        if rcg_details['name'] != new_name:
            if not self.module.check_mode:
                self.powerflex_conn.replication_consistency_group.rename_rcg(rcg_id, new_name)
            return True
    except Exception as e:
        errormsg = 'Renaming replication consistency group to {0} failed with error {1}'.format(new_name, str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)