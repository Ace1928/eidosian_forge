from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, json
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import MerakiModule, meraki_argument_spec
def normalize_rule_case(rules):
    excluded = ['comment']
    try:
        for r in rules['rules']:
            for k in r:
                if k not in excluded:
                    r[k] = r[k].lower()
    except KeyError:
        return rules
    return rules