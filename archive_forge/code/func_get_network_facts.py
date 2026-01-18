from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible.module_utils.common.text.formatters import bytes_to_human
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
def get_network_facts(self):
    facts = dict()
    facts['ansible_interfaces'] = []
    facts['ansible_all_ipv4_addresses'] = []
    for nic in self.host.config.network.vnic:
        device = nic.device
        facts['ansible_interfaces'].append(device)
        facts['ansible_all_ipv4_addresses'].append(nic.spec.ip.ipAddress)
        _tmp = {'device': device, 'ipv4': {'address': nic.spec.ip.ipAddress, 'netmask': nic.spec.ip.subnetMask}, 'macaddress': nic.spec.mac, 'mtu': nic.spec.mtu}
        facts['ansible_' + device] = _tmp
    return facts