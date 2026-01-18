from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
def gather_host_firewall_info(self):
    hosts_firewall_info = dict()
    for host in self.hosts:
        firewall_system = host.configManager.firewallSystem
        if firewall_system:
            hosts_firewall_info[host.name] = []
            for rule_set_obj in firewall_system.firewallInfo.ruleset:
                hosts_firewall_info[host.name].append(self.normalize_rule_set(rule_obj=rule_set_obj))
    return hosts_firewall_info