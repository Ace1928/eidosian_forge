from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import MerakiModule, meraki_argument_spec
import json
def normalize_dhcp_handling(parameter):
    if parameter == 'none':
        return 'Do not respond to DHCP requests'
    elif parameter == 'server':
        return 'Run a DHCP server'
    elif parameter == 'relay':
        return 'Relay DHCP to another server'