from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, json
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import MerakiModule, meraki_argument_spec
def compare_rule_count(original, payload):
    if len(original['rules']) - 1 != len(payload['rules']):
        return True
    return False