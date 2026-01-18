from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
def create_acr(client_obj, state, initiator_group, volume, **kwargs):
    if utils.is_null_or_empty(initiator_group):
        return (False, False, 'Access control record creation failed. No initiator group provided.', {})
    if utils.is_null_or_empty(volume):
        return (False, False, 'Access control record creation failed. No volume name provided.', {})
    try:
        ig_resp = client_obj.initiator_groups.get(id=None, name=initiator_group)
        if ig_resp is None:
            return (False, False, f"Initiator Group '{initiator_group}' is not present on array.", {})
        vol_resp = client_obj.volumes.get(id=None, name=volume)
        if vol_resp is None:
            return (False, False, f"Volume name '{volume}' is not present on array.", {})
        acr_resp = client_obj.access_control_records.get(vol_name=volume, initiator_group_name=initiator_group, apply_to=kwargs['apply_to'])
        if utils.is_null_or_empty(acr_resp) is False:
            changed_attrs_dict, params = utils.remove_unchanged_or_null_args(acr_resp, **kwargs)
        else:
            params = utils.remove_null_args(**kwargs)
        if acr_resp is None or changed_attrs_dict.__len__() > 0:
            acr_resp = client_obj.access_control_records.create(initiator_group_id=ig_resp.attrs.get('id'), vol_id=vol_resp.attrs.get('id'), **params)
            return (True, True, 'Successfully created access control record.', acr_resp.attrs)
        elif state == 'present':
            return (True, False, f"Access control record for volume '{volume}' with initiator group '{initiator_group}' is already present.", acr_resp.attrs)
        return (False, False, f"Access control record for volume '{volume}' with initiator group '{initiator_group}' cannot be created as it is already present.", {})
    except Exception as ex:
        return (False, False, f'Access control record creation failed | {ex}', {})