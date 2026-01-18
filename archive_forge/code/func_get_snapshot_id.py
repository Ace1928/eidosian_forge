from __future__ import absolute_import, division, print_function
import datetime
import uuid
def get_snapshot_id(client_obj, vol_name, snap_name):
    if is_null_or_empty(vol_name) or is_null_or_empty(snap_name):
        return None
    else:
        resp = client_obj.snapshots.get(vol_name=vol_name, name=snap_name)
        if resp is None:
            raise Exception(f"No snapshot with name '{snap_name}' found for volume {vol_name}.")
        return resp.attrs.get('id')