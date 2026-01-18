from __future__ import (absolute_import, division, print_function)
import os
import traceback
from ansible.module_utils.basic import missing_required_lib
def query_profile_policies(self, profile_id):
    """ Returns a set of the policy objects assigned to the resource
        """
    url = '{api_url}/policy_profiles/{profile_id}?expand=policies'
    try:
        response = self.client.get(url.format(api_url=self.api_url, profile_id=profile_id))
    except Exception as e:
        msg = 'Failed to query {resource_type} policies: {error}'.format(resource_type=self.resource_type, error=e)
        self.module.fail_json(msg=msg)
    resources = response.get('policies', [])
    policies = [self.clean_policy_object(policy) for policy in resources]
    return policies