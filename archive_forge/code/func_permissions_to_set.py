from __future__ import absolute_import, division, print_function
import uuid
from ansible.module_utils._text import to_native
def permissions_to_set(self, permissions):
    new_permissions = [str(dict(actions=set([to_native(a) for a in item.get('actions')]) if item.get('actions') else None, not_actions=set([to_native(a) for a in item.get('not_actions')]) if item.get('not_actions') else None, data_actions=set([to_native(a) for a in item.get('data_actions')]) if item.get('data_actions') else None, not_data_actions=set([to_native(a) for a in item.get('not_data_actions')]) if item.get('not_data_actions') else None)) for item in permissions]
    return set(new_permissions)