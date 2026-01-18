from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
import copy
def check_ip_in_secondarys(self, standby_ip, cluster_details):
    """whether standby IPs present in secondary MDMs"""
    if 'slaves' in cluster_details:
        for secondary_mdm in cluster_details['slaves']:
            current_secondary_ips = secondary_mdm['ips']
            for ips in standby_ip:
                if ips in current_secondary_ips:
                    LOG.info(self.exist_msg)
                    return False
    return True