from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
import copy
def check_ip_in_tbs(self, standby_ip, cluster_details):
    """whether standby IPs present in tie-breaker MDMs"""
    if 'tieBreakers' in cluster_details:
        for tb_mdm in cluster_details['tieBreakers']:
            current_tb_ips = tb_mdm['ips']
            for ips in standby_ip:
                if ips in current_tb_ips:
                    LOG.info(self.exist_msg)
                    return False
    return True