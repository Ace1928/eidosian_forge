from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible.module_utils._text import to_native
def check_drs_config_diff(self):
    """
        Check DRS configuration diff
        Returns: True if there is diff, else False
        """
    drs_config = self.cluster.configurationEx.drsConfig
    if drs_config.enabled != self.enable_drs or drs_config.enableVmBehaviorOverrides != self.params.get('drs_enable_vm_behavior_overrides') or drs_config.defaultVmBehavior != self.params.get('drs_default_vm_behavior') or (drs_config.vmotionRate != self.drs_vmotion_rate) or (self.cluster.configurationEx.proactiveDrsConfig.enabled != self.params.get('predictive_drs')):
        return True
    if self.changed_advanced_settings:
        return True
    return False