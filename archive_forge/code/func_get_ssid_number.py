from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, json
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import MerakiModule, meraki_argument_spec
def get_ssid_number(name, data):
    for ssid in data:
        if name == ssid['name']:
            return ssid['number']
    return False