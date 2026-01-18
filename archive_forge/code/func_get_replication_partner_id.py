from __future__ import absolute_import, division, print_function
import datetime
import uuid
def get_replication_partner_id(client_obj, replication_partner_name):
    if is_null_or_empty(replication_partner_name):
        return None
    else:
        resp = client_obj.replication_partners.get(name=replication_partner_name)
        if resp is None:
            raise Exception(f'Invalid value for replication partner {replication_partner_name}')
        return resp.attrs.get('id')