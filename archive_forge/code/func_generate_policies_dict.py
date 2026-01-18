from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
from datetime import datetime
def generate_policies_dict(blade):
    policies_info = {}
    policies = blade.policies.list_policies()
    for policycnt in range(0, len(policies.items)):
        policy = policies.items[policycnt].name
        policies_info[policy] = {}
        policies_info[policy]['enabled'] = policies.items[policycnt].enabled
        if policies.items[policycnt].rules:
            policies_info[policy]['rules'] = policies.items[policycnt].rules[0].to_dict()
    return policies_info