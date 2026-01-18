from __future__ import absolute_import, division, print_function
import datetime
import uuid
def get_pe_id(client_obj, pe_name):
    if is_null_or_empty(pe_name):
        return None
    else:
        resp = client_obj.protocol_endpoints.get(name=pe_name)
        if resp is None:
            raise Exception(f'Invalid value for protection endpoint {pe_name}')
        return resp.attrs.get('id')