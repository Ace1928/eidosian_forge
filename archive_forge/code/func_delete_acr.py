from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
def delete_acr(client_obj, initiator_group, volume, **kwargs):
    if utils.is_null_or_empty(initiator_group):
        return (False, False, 'Access control record deletion failed. No initiator group provided.')
    if utils.is_null_or_empty(volume):
        return (False, False, 'Access control record deletion failed. No volume provided.')
    params = utils.remove_null_args(**kwargs)
    try:
        acr_list_resp = client_obj.access_control_records.list(vol_name=volume, initiator_group_name=initiator_group, **params)
        if acr_list_resp is not None and acr_list_resp.__len__() > 0:
            for acr_resp in acr_list_resp:
                client_obj.access_control_records.delete(acr_resp.attrs.get('id'))
            return (True, True, f"Successfully deleted access control record for initiator group '{initiator_group}' associated with volume '{volume}'.")
        else:
            return (True, False, f"No access control record for initiator group '{initiator_group}' associated with volume '{volume}' found.")
    except Exception as ex:
        return (False, False, f'Access control record deletion failed | {ex}')