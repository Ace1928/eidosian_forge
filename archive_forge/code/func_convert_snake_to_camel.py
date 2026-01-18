from __future__ import absolute_import, division, print_function
import time
import re
from ansible.module_utils.basic import json, env_fallback
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict, snake_dict_to_camel_dict, recursive_diff
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils._text import to_native
def convert_snake_to_camel(self, data):
    """
        Converts a dictionary or list to camel case from snake case
        :type data: dict or list
        :return: Converted data structure, if list or dict
        """
    if isinstance(data, dict):
        return snake_dict_to_camel_dict(data)
    elif isinstance(data, list):
        return [snake_dict_to_camel_dict(item) for item in data]
    else:
        return data