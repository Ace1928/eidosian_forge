from __future__ import absolute_import, division, print_function
import time
import re
from ansible.module_utils.basic import json, env_fallback
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict, snake_dict_to_camel_dict, recursive_diff
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils._text import to_native
def generate_diff(self, before, after):
    """Creates a diff based on two objects. Applies to the object and returns nothing.
        """
    try:
        diff = recursive_diff(before, after)
        self.result['diff'] = {'before': diff[0], 'after': diff[1]}
    except AttributeError:
        diff = recursive_diff({'data': before}, {'data': after})
        self.result['diff'] = {'before': diff[0]['data'], 'after': diff[1]['data']}