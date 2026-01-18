from __future__ import absolute_import, division, print_function
import time
import re
from ansible.module_utils.basic import json, env_fallback
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict, snake_dict_to_camel_dict, recursive_diff
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils._text import to_native
def get_net(self, org_name, net_name=None, org_id=None, data=None, net_id=None):
    """ Return network information """
    if not data:
        if not org_id:
            org_id = self.get_org_id(org_name)
        data = self.get_nets(org_id=org_id)
    for n in data:
        if net_id:
            if n['id'] == net_id:
                return n
        elif net_name:
            if n['name'] == net_name:
                return n
    return False