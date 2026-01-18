from __future__ import absolute_import, division, print_function
import time
import re
from ansible.module_utils.basic import json, env_fallback
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict, snake_dict_to_camel_dict, recursive_diff
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils._text import to_native
def is_org_valid(self, data, org_name=None, org_id=None):
    """Checks whether a specific org exists and is duplicated.

        If 0, doesn't exist. 1, exists and not duplicated. >1 duplicated.
        """
    org_count = 0
    if org_name is not None:
        for o in data:
            if o['name'] == org_name:
                org_count += 1
    if org_id is not None:
        for o in data:
            if o['id'] == org_id:
                org_count += 1
    return org_count