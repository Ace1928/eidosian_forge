from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
import copy
def check_mdm_exists(self, standby_ip=None, cluster_details=None):
    """Check whether standby MDM exists in MDM Cluster"""
    current_master_ips = cluster_details['master']['ips']
    for ips in standby_ip:
        if ips in current_master_ips:
            LOG.info(self.exist_msg)
            return False
    in_secondary = self.check_ip_in_secondarys(standby_ip=standby_ip, cluster_details=cluster_details)
    if not in_secondary:
        return False
    in_tbs = self.check_ip_in_tbs(standby_ip=standby_ip, cluster_details=cluster_details)
    if not in_tbs:
        return False
    in_standby = self.check_ip_in_standby(standby_ip=standby_ip, cluster_details=cluster_details)
    if not in_standby:
        return False
    LOG.info('New Standby MDM does not exists in MDM cluster')
    return True