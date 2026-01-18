from __future__ import absolute_import, division, print_function
import time
import re
from ansible.module_utils.basic import json, env_fallback
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict, snake_dict_to_camel_dict, recursive_diff
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils._text import to_native
def get_net_id(self, org_name=None, net_name=None, data=None):
    """Return network id from lookup or existing data."""
    if data is None:
        self.fail_json(msg='Must implement lookup')
    for n in data:
        if n['name'] == net_name:
            return n['id']
    self.fail_json(msg='No network found with the name {0}'.format(net_name))