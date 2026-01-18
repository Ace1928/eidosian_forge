from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
def group_encryption(client_obj, group_name, encryption_config):
    if utils.is_null_or_empty(group_name):
        return (False, False, 'Encryption setting for group failed as group name is not present.', {}, {})
    try:
        group_resp = client_obj.groups.get(id=None, name=group_name)
        if utils.is_null_or_empty(group_resp):
            return (False, False, f"Encryption setting for group '{group_name}' cannot be done as it is not present.", {}, {})
        changed_attrs_dict, params = utils.remove_unchanged_or_null_args(group_resp, encryption_config=encryption_config)
        if changed_attrs_dict.__len__() > 0:
            group_resp = client_obj.groups.update(id=group_resp.attrs.get('id'), encryption_config=encryption_config)
            return (True, True, f"Encryption setting for group '{group_name}' changed successfully. ", changed_attrs_dict, group_resp.attrs)
        else:
            return (True, False, f"Encryption setting for group '{group_resp.attrs.get('name')}' is already in same state.", {}, group_resp.attrs)
    except Exception as ex:
        return (False, False, f'Encryption setting for group failed |{ex}', {}, {})