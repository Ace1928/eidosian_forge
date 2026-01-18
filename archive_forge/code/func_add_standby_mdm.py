from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
import copy
def add_standby_mdm(self, mdm_name, standby_mdm, cluster_details):
    """ Adding a standby MDM"""
    if self.module.params['state'] == 'present' and standby_mdm is not None and self.check_mdm_exists(standby_mdm['mdm_ips'], cluster_details):
        self.is_id_new_name_in_add_mdm()
        mdm_details = self.is_mdm_name_id_exists(mdm_name=mdm_name, cluster_details=cluster_details)
        if mdm_details:
            LOG.info('Standby MDM %s exits in the system', mdm_name)
            return (False, cluster_details)
        standby_payload = prepare_standby_payload(standby_mdm)
        standby_add = self.perform_add_standby(mdm_name, standby_payload)
        if standby_add:
            cluster_details = self.get_mdm_cluster_details()
            msg = 'Fetched the MDM cluster details {0} after adding a standby MDM'.format(str(cluster_details))
            LOG.info(msg)
            return (True, cluster_details)
    return (False, cluster_details)