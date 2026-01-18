from __future__ import (absolute_import, division, print_function)
import os
import json
import traceback
from ansible.module_utils.basic import env_fallback
def ecs_connect(module):
    """ Return an ecs connection"""
    ecs_params = get_profile(module.params)
    region = module.params.get('alicloud_region')
    if region:
        try:
            ecs = connect_to_acs(footmark.ecs, region, **ecs_params)
        except AnsibleACSError as e:
            module.fail_json(msg=str(e))
    return ecs