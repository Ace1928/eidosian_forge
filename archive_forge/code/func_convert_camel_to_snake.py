from __future__ import absolute_import, division, print_function
import time
import re
from ansible.module_utils.basic import json, env_fallback
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict, snake_dict_to_camel_dict, recursive_diff
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils._text import to_native
def convert_camel_to_snake(self, data):
    """
        Converts a dictionary or list to snake case from camel case
        :type data: dict or list
        :return: Converted data structure, if list or dict
        """
    if isinstance(data, dict):
        return camel_dict_to_snake_dict(data, ignore_list=('tags', 'tag'))
    elif isinstance(data, list):
        return [camel_dict_to_snake_dict(item, ignore_list=('tags', 'tag')) for item in data]
    else:
        return data