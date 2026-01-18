from __future__ import absolute_import, division, print_function
import time
import re
from ansible.module_utils.basic import json, env_fallback
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict, snake_dict_to_camel_dict, recursive_diff
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils._text import to_native
def get_nets(self, org_name=None, org_id=None):
    """Downloads all networks in an organization."""
    if org_name:
        org_id = self.get_org_id(org_name)
    path = self.construct_path('get_all', org_id=org_id, function='network', params={'perPage': '1000'})
    r = self.request(path, method='GET', pagination_items=1000)
    if self.status != 200:
        self.fail_json(msg='Network lookup failed')
    self.nets = r
    templates = self.get_config_templates(org_id)
    for t in templates:
        self.nets.append(t)
    return self.nets