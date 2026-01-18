from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
import ipaddress
from ipaddress import ip_network
def get_interface_parameters():
    """This method provide parameters required for the ansible
       Interface module on Unity"""
    return dict(nas_server_id=dict(type='str'), nas_server_name=dict(type='str'), ethernet_port_name=dict(type='str'), ethernet_port_id=dict(type='str'), role=dict(type='str', choices=['PRODUCTION', 'BACKUP']), interface_ip=dict(required=True, type='str'), netmask=dict(type='str'), prefix_length=dict(type='int'), gateway=dict(type='str'), vlan_id=dict(type='int'), state=dict(required=True, type='str', choices=['present', 'absent']))