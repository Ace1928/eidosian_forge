from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def convert_user_group_list_of_str_to_list_of_dict(self, groups):
    list_of_groups = []
    if isinstance(groups, list) and len(groups) > 0:
        for group in groups:
            if isinstance(group, str):
                group_dict = {}
                group_dict['name'] = group
                list_of_groups.append(group_dict)
    return list_of_groups