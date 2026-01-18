from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
import copy
def remove_standby_mdm(self, mdm_name, mdm_id, cluster_details):
    """ Remove the Standby MDM
        :param mdm_id: ID of MDM that will become owner of MDM cluster
        :param mdm_name: Name of MDM that will become owner of MDM cluster
        :param cluster_details: Details of MDM cluster
        :return: True if MDM removed successful
        """
    name_or_id = mdm_id if mdm_id else mdm_name
    if mdm_id is None and mdm_name is None:
        err_msg = 'Either mdm_name or mdm_id is required while removing the standby MDM.'
        LOG.error(err_msg)
        self.module.fail_json(msg=err_msg)
    mdm_details = self.is_mdm_name_id_exists(mdm_name=mdm_name, mdm_id=mdm_id, cluster_details=cluster_details)
    if mdm_details is None:
        LOG.info('MDM %s not exists in MDM cluster.', name_or_id)
        return False
    mdm_id = mdm_details['id']
    try:
        if not self.module.check_mode:
            self.powerflex_conn.system.remove_standby_mdm(mdm_id=mdm_id)
        return True
    except Exception as e:
        error_msg = 'Failed to remove the standby MDM {0} from the MDM cluster with error {1}'.format(name_or_id, str(e))
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)