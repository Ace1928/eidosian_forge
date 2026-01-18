from __future__ import absolute_import, division, print_function
import datetime
import uuid
def get_vol_id(client_obj, vol_name):
    if is_null_or_empty(vol_name):
        return None
    else:
        resp = client_obj.volumes.get(name=vol_name)
        if resp is None:
            raise Exception(f'Invalid value for volume {vol_name}')
        return resp.attrs.get('id')