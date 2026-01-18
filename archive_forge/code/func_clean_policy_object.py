from __future__ import (absolute_import, division, print_function)
import os
import traceback
from ansible.module_utils.basic import missing_required_lib
def clean_policy_object(self, policy):
    """ Clean a policy object to have human readable form of:
        {
            name: STR,
            description: STR,
            active: BOOL
        }
        """
    name = policy.get('name')
    description = policy.get('description')
    active = policy.get('active')
    return dict(name=name, description=description, active=active)