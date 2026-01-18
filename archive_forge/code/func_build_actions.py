from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
def build_actions(actions):
    action_items = []
    for action in actions:
        action_item = snake_dict_to_camel_dict(action)
        action_items.append(action_item)
    return action_items