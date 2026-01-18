from __future__ import absolute_import, division, print_function
import datetime
import uuid
def get_owned_by_group_id(client_obj, owned_by_group_name):
    if is_null_or_empty(owned_by_group_name):
        return None
    else:
        resp = client_obj.groups.get(name=owned_by_group_name)
        if resp is None:
            raise Exception(f'Invalid value for owned by group {owned_by_group_name}')
        return resp.attrs.get('id')