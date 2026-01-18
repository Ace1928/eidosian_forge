from __future__ import absolute_import, division, print_function
import datetime
import uuid
def get_initiator_group_id(client_obj, ig_name):
    if is_null_or_empty(ig_name):
        return None
    else:
        resp = client_obj.initiator_groups.get(name=ig_name)
        if resp is None:
            raise Exception(f'Invalid value for initiator group {ig_name}')
        return resp.attrs.get('id')