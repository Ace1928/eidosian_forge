from __future__ import (absolute_import, division, print_function)
import os
import json
import traceback
from ansible.module_utils.basic import env_fallback
def connect_to_acs(acs_module, region, **params):
    conn = acs_module.connect_to_region(region, **params)
    if not conn:
        if region not in [acs_module_region.id for acs_module_region in acs_module.regions()]:
            raise AnsibleACSError('Region %s does not seem to be available for acs module %s.' % (region, acs_module.__name__))
        else:
            raise AnsibleACSError('Unknown problem connecting to region %s for acs module %s.' % (region, acs_module.__name__))
    return conn