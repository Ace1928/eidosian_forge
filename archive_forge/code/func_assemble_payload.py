from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, json
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import MerakiModule, meraki_argument_spec
def assemble_payload(meraki):
    params_map = {'policy': 'policy', 'protocol': 'protocol', 'dest_port': 'destPort', 'dest_cidr': 'destCidr', 'comment': 'comment'}
    rules = []
    for rule in meraki.params['rules']:
        proposed_rule = dict()
        for k, v in rule.items():
            proposed_rule[params_map[k]] = v
        rules.append(proposed_rule)
    payload = {'rules': rules}
    return payload