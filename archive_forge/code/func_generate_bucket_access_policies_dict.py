from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
from datetime import datetime
def generate_bucket_access_policies_dict(blade):
    policies_info = {}
    policies = list(blade.get_buckets_bucket_access_policies().items)
    for policy in range(0, len(policies)):
        policy_name = policies[policy].name
        policies_info[policy_name] = {'description': policies[policy].description, 'enabled': policies[policy].enabled, 'local': policies[policy].is_local, 'rules': []}
        for rule in range(0, len(policies[policy].rules)):
            policies_info[policy_name]['rules'].append({'actions': policies[policy].rules[rule].actions, 'resources': policies[policy].rules[rule].resources, 'all_principals': policies[policy].rules[rule].principals.all, 'effect': policies[policy].rules[rule].effect, 'name': policies[policy].rules[rule].name})
    return policies_info