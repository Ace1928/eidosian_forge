from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
from datetime import datetime
def generate_bucket_cross_object_policies_dict(blade):
    policies_info = {}
    policies = list(blade.get_buckets_cross_origin_resource_sharing_policies().items)
    for policy in range(0, len(policies)):
        policy_name = policies[policy].name
        policies_info[policy_name] = {'allowed_headers': policies[policy].allowed_headers, 'allowed_methods': policies[policy].allowed_methods, 'allowed_origins': policies[policy].allowed_origins}
    return policies_info