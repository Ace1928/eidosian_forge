from __future__ import absolute_import, division, print_function
import datetime
import uuid
def get_chap_user_id(client_obj, chap_user_name):
    if is_null_or_empty(chap_user_name):
        return None
    else:
        resp = client_obj.chap_users.get(name=chap_user_name)
        if resp is None:
            raise Exception(f'Invalid value for chap user {chap_user_name}')
        return resp.attrs.get('id')