from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
import copy
def change_ownership(self, mdm_id=None, mdm_name=None, cluster_details=None):
    """ Change the ownership of MDM cluster.
        :param mdm_id: ID of MDM that will become owner of MDM cluster
        :param mdm_name: Name of MDM that will become owner of MDM cluster
        :param cluster_details: Details of MDM cluster
        :return: True if Owner of MDM cluster change successful
        """
    name_or_id = mdm_id if mdm_id else mdm_name
    if mdm_id is None and mdm_name is None:
        err_msg = 'Either mdm_name or mdm_id is required while changing ownership of MDM cluster.'
        LOG.error(err_msg)
        self.module.fail_json(msg=err_msg)
    mdm_details = self.is_mdm_name_id_exists(mdm_name=mdm_name, mdm_id=mdm_id, cluster_details=cluster_details)
    if mdm_details is None:
        err_msg = self.not_exist_msg.format(name_or_id)
        self.module.fail_json(msg=err_msg)
    mdm_id = mdm_details['id']
    if mdm_details['id'] == cluster_details['master']['id']:
        LOG.info('MDM %s is already Owner of MDM cluster.', name_or_id)
        return False
    else:
        try:
            if not self.module.check_mode:
                self.powerflex_conn.system.change_mdm_ownership(mdm_id=mdm_id)
            return True
        except Exception as e:
            error_msg = 'Failed to update the Owner of MDM cluster to MDM {0} with error {1}'.format(name_or_id, str(e))
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)