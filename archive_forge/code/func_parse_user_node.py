from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils._text import to_bytes
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import string_types
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.argspec.logging_global.logging_global import (
def parse_user_node(self, conf):
    users_list = []
    users = []
    if isinstance(conf, dict):
        users.append(conf)
    else:
        users = conf
    for user in users:
        user_dict = {}
        user_dict['name'] = user.get('name')
        if 'allow-duplicates' in user:
            user_dict['allow_duplicates'] = True
        if 'contents' in user.keys():
            contents = user.get('contents')
            if isinstance(contents, list):
                for content in contents:
                    user_dict = self.parse_console_node(content, user_dict)
            else:
                user_dict = self.parse_console_node(contents, user_dict)
        if 'match' in user.keys():
            user_dict['match'] = user.get('match')
        if 'match-strings' in user.keys():
            match_strings = user.get('match-strings')
            match_strings_list = []
            if isinstance(match_strings, list):
                for item in match_strings:
                    match_strings_list.append(item)
            else:
                match_strings_list.append(match_strings)
            user_dict['match_strings'] = match_strings_list
        users_list.append(user_dict)
    return users_list