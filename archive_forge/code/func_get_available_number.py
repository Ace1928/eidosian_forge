from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, json
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import (
def get_available_number(data):
    for item in data:
        if 'Unconfigured SSID' in item['name']:
            return item['number']
    return False