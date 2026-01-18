from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
from ansible.module_utils._text import to_native
import socket
def gather_rule_set(self):
    for host in self.hosts:
        self.firewall_facts[host.name] = {}
        firewall_system = host.configManager.firewallSystem
        if firewall_system:
            for rule_set_obj in firewall_system.firewallInfo.ruleset:
                temp_rule_dict = dict()
                temp_rule_dict['enabled'] = rule_set_obj.enabled
                allowed_host = rule_set_obj.allowedHosts
                rule_allow_host = dict()
                rule_allow_host['ip_address'] = allowed_host.ipAddress
                rule_allow_host['ip_network'] = [ip.network + '/' + str(ip.prefixLength) for ip in allowed_host.ipNetwork]
                rule_allow_host['all_ip'] = allowed_host.allIp
                temp_rule_dict['allowed_hosts'] = rule_allow_host
                self.firewall_facts[host.name][rule_set_obj.key] = temp_rule_dict