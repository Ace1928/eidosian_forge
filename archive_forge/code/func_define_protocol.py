from __future__ import absolute_import, division, print_function
import time
import re
from ansible.module_utils.basic import json, env_fallback
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict, snake_dict_to_camel_dict, recursive_diff
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils._text import to_native
def define_protocol(self):
    """Set protocol based on use_https parameters."""
    if self.params['use_https'] is True:
        self.params['protocol'] = 'https'
    else:
        self.params['protocol'] = 'http'