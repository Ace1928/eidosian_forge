from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def check_vmk_current_state(self):
    self.host_system = find_hostsystem_by_name(self.content, self.esxi_hostname)
    for vnic in self.host_system.configManager.networkSystem.networkInfo.vnic:
        if vnic.device == self.device:
            if vnic.spec.distributedVirtualPort is None:
                std_vswitches = [vswitch.name for vswitch in self.host_system.configManager.networkSystem.networkInfo.vswitch]
                if self.current_switch_name not in std_vswitches:
                    return 'migrated'
                if vnic.portgroup == self.current_portgroup_name:
                    return 'migrate_vss_vds'
            else:
                dvs = find_dvs_by_name(self.content, self.current_switch_name)
                if dvs is None:
                    return 'migrated'
                if vnic.spec.distributedVirtualPort.switchUuid == dvs.uuid:
                    return 'migrate_vds_vss'
    self.module.fail_json(msg='Unable to find the specified device %s.' % self.device)