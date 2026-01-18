from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
from enum import Enum
def create_clone_from_snapshot(client_obj, snap_list_resp, vol_name, snapshot_to_clone, state):
    if utils.is_null_or_empty(client_obj) or utils.is_null_or_empty(vol_name) or utils.is_null_or_empty(snap_list_resp) or utils.is_null_or_empty(snapshot_to_clone):
        return (False, 'Create clone from snapshot failed as valid arguments are not provided. Please check the argument provided for volume and snapshot.', {})
    try:
        for snap_obj in snap_list_resp:
            if snap_obj.attrs.get('name') == snapshot_to_clone:
                resp = client_obj.volumes.create(name=vol_name, base_snap_id=snap_obj.attrs.get('id'), clone=True)
                if utils.is_null_or_empty(resp) is False:
                    return (Vol_Operation.SUCCESS, f'{vol_name}', resp.attrs)
        return Vol_Operation.FAILED
    except exceptions.NimOSAPIError as ex:
        if 'SM_eexist' in str(ex):
            if state == 'present':
                return (Vol_Operation.ALREADY_EXISTS, f"Cloned volume '{vol_name}' is already present in given state.", {})
            else:
                return (Vol_Operation.FAILED, f"Create clone from snapshot failed as cloned volume '{vol_name}' already exist| {ex}", {})
    except Exception as ex:
        return (Vol_Operation.FAILED, f'Create clone from snapshot failed | {ex}', {})