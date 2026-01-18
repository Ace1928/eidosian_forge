from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
def gather_vswitch_info(self):
    """Gather vSwitch info"""
    hosts_vswitch_info = dict()
    for host in self.hosts:
        network_manager = host.configManager.networkSystem
        if network_manager:
            temp_switch_dict = dict()
            for vswitch in network_manager.networkInfo.vswitch:
                temp_switch_dict[vswitch.name] = self.normalize_vswitch_info(vswitch_obj=vswitch, policy_info=self.policies)
            hosts_vswitch_info[host.name] = temp_switch_dict
    return hosts_vswitch_info