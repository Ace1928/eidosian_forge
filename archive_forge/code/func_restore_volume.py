from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
from enum import Enum
def restore_volume(client_obj, vol_name, snapshot_to_restore=None):
    if utils.is_null_or_empty(vol_name):
        return (False, False, 'Volume restore failed as volume name is null.', {}, {})
    try:
        vol_resp = client_obj.volumes.get(id=None, name=vol_name)
        if utils.is_null_or_empty(vol_resp):
            return (False, False, f"Volume '{vol_name}' not present to restore.", {}, {})
        if utils.is_null_or_empty(snapshot_to_restore):
            snap_list_resp = client_obj.snapshots.list(vol_name=vol_name)
            if utils.is_null_or_empty(snap_list_resp):
                return (False, False, f"Volume '{vol_name}' cannot be restored as no snapshot is present in source volume.", {}, {})
            snap_resp = snap_list_resp[-1]
            snapshot_to_restore = snap_resp.attrs.get('name')
        else:
            snap_resp = client_obj.snapshots.get(vol_name=vol_name, name=snapshot_to_restore)
            if utils.is_null_or_empty(snap_resp):
                return (False, False, f"Volume '{vol_name}' cannot not be restored as given snapshot name '{snapshot_to_restore}' is not present insource volume.", {}, {})
        client_obj.volumes.offline(id=vol_resp.attrs.get('id'))
        resp = client_obj.volumes.restore(base_snap_id=snap_resp.attrs.get('id'), id=vol_resp.attrs.get('id'))
        client_obj.volumes.online(id=vol_resp.attrs.get('id'))
        return (True, True, f"Restored volume '{vol_name}' from snapshot '{snapshot_to_restore}' successfully.", {}, resp)
    except Exception as ex:
        return (False, False, f"Volume restore failed '{ex}'", {}, {})