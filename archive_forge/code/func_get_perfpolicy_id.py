from __future__ import absolute_import, division, print_function
import datetime
import uuid
def get_perfpolicy_id(client_obj, perfpolicy_name):
    if is_null_or_empty(perfpolicy_name):
        return None
    else:
        resp = client_obj.performance_policies.get(name=perfpolicy_name)
        if resp is None:
            raise Exception(f'Invalid value for performance policy: {perfpolicy_name}')
        return resp.attrs.get('id')