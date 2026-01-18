from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.argspec.l2_interfaces.l2_interfaces import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.l2_interfaces import (
def process_mode(obj):
    mode = ''
    if obj == 'dot1q-tunnel':
        mode = 'dot1q_tunnel'
    elif obj == 'dynamic auto':
        mode = 'dynamic_auto'
    elif obj == 'dynamic desirable':
        mode = 'dynamic_desirable'
    elif obj == 'private-vlan host':
        mode = 'private_vlan_host'
    elif obj == 'private-vlan promiscuous':
        mode = 'private_vlan_promiscuous'
    elif obj == 'private-vlan trunk secondary':
        mode = 'private_vlan_trunk'
    return mode