from __future__ import absolute_import, division, print_function
import datetime
import uuid
def get_pool_id(client_obj, pool_name):
    if is_null_or_empty(pool_name):
        return None
    else:
        resp = client_obj.pools.get(name=pool_name)
        if resp is None:
            raise Exception(f'Invalid value for pool {pool_name}')
        return resp.attrs.get('id')