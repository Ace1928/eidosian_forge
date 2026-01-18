from __future__ import absolute_import, division, print_function
import datetime
import uuid
def get_folder_id(client_obj, folder_name):
    if is_null_or_empty(folder_name):
        return None
    else:
        resp = client_obj.folders.get(name=folder_name)
        if resp is None:
            raise Exception(f'Invalid value for folder {folder_name}')
        return resp.attrs.get('id')