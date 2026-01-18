from __future__ import absolute_import, division, print_function
import datetime
import uuid
def get_volcoll_id(client_obj, volcoll_name):
    if is_null_or_empty(volcoll_name):
        return None
    else:
        resp = client_obj.volume_collections.get(name=volcoll_name)
        if resp is None:
            raise Exception(f'Invalid value for volcoll {volcoll_name}')
        return resp.attrs.get('id')