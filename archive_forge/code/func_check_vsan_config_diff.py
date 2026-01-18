from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible.module_utils._text import to_native
def check_vsan_config_diff(self):
    """
        Check VSAN configuration diff
        Returns: True if there is diff, else False

        """
    vsan_config = self.cluster.configurationEx.vsanConfigInfo
    if vsan_config.enabled != self.enable_vsan or vsan_config.defaultConfig.autoClaimStorage != self.params.get('vsan_auto_claim_storage'):
        return True
    if self.advanced_options is not None:
        vsan_config_info = self.vsanClusterConfigSystem.GetConfigInfoEx(self.cluster).extendedConfig
        if self.advanced_options['automatic_rebalance'] is not None and self.advanced_options['automatic_rebalance'] != vsan_config_info.proactiveRebalanceInfo.enabled:
            return True
        if self.advanced_options['disable_site_read_locality'] is not None and self.advanced_options['disable_site_read_locality'] != vsan_config_info.disableSiteReadLocality:
            return True
        if self.advanced_options['large_cluster_support'] is not None and self.advanced_options['large_cluster_support'] != vsan_config_info.largeScaleClusterSupport:
            return True
        if self.advanced_options['object_repair_timer'] is not None and self.advanced_options['object_repair_timer'] != vsan_config_info.objectRepairTimer:
            return True
        if self.advanced_options['thin_swap'] is not None and self.advanced_options['thin_swap'] != vsan_config_info.enableCustomizedSwapObject:
            return True
    return False