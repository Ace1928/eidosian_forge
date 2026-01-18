from __future__ import (absolute_import, division, print_function)
import os
import traceback
from ansible.module_utils.basic import missing_required_lib
def clean_profile_object(self, profile):
    """ Clean a profile object to have human readable form of:
        {
            profile_name: STR,
            profile_description: STR,
            policies: ARR<POLICIES>
        }
        """
    profile_id = profile['id']
    name = profile.get('name')
    description = profile.get('description')
    policies = self.query_profile_policies(profile_id)
    return dict(profile_name=name, profile_description=description, policies=policies)