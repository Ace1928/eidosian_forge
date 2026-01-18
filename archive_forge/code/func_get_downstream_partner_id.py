from __future__ import absolute_import, division, print_function
import datetime
import uuid
def get_downstream_partner_id(client_obj, downstream_partner):
    if is_null_or_empty(downstream_partner):
        return None
    else:
        resp = client_obj.replication_partners.get(name=downstream_partner)
        if resp is None:
            raise Exception(f'Invalid value for downstream partner {downstream_partner}')
        return resp.attrs.get('id')