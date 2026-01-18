from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
import copy
def change_cluster_mode(self, cluster_mode, mdm, cluster_details):
    """change the MDM cluster mode.
        :param cluster_mode: specifies the mode of MDM cluster
        :param mdm: A dict containing parameters to change MDM cluster mode
        :param cluster_details: Details of MDM cluster
        :return: True if mode changed successfully
        """
    self.is_none_name_id_in_switch_cluster_mode(mdm=mdm)
    if cluster_mode == cluster_details['clusterMode']:
        LOG.info('MDM cluster is already in required mode.')
        return False
    add_secondary = []
    add_tb = []
    remove_secondary = []
    remove_tb = []
    if self.module.params['state'] == 'present' and self.module.params['mdm_state'] == 'present-in-cluster':
        add_secondary, add_tb = self.cluster_expand_list(mdm, cluster_details)
    elif self.module.params['state'] == 'present' and self.module.params['mdm_state'] == 'absent-in-cluster':
        remove_secondary, remove_tb = self.cluster_reduce_list(mdm, cluster_details)
    try:
        if not self.module.check_mode:
            self.powerflex_conn.system.switch_cluster_mode(cluster_mode=cluster_mode, add_secondary=add_secondary, remove_secondary=remove_secondary, add_tb=add_tb, remove_tb=remove_tb)
        return True
    except Exception as e:
        err_msg = 'Failed to change the MDM cluster mode with error {0}'.format(str(e))
        LOG.error(err_msg)
        self.module.fail_json(msg=err_msg)