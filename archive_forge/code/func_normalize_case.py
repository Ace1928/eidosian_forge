from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, json
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import MerakiModule, meraki_argument_spec
def normalize_case(rule):
    any = ['any', 'Any', 'ANY']
    if 'srcPort' in rule:
        if rule['srcPort'] in any:
            rule['srcPort'] = 'Any'
    if 'srcCidr' in rule:
        if rule['srcCidr'] in any:
            rule['srcCidr'] = 'Any'
    if 'destPort' in rule:
        if rule['destPort'] in any:
            rule['destPort'] = 'Any'
    if 'destCidr' in rule:
        if rule['destCidr'] in any:
            rule['destCidr'] = 'Any'