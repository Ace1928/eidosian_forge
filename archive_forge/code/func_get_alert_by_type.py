from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, json
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import (
def get_alert_by_type(type, meraki):
    for alert in meraki.params['alerts']:
        if alert['alert_type'] == type:
            return alert
    return None