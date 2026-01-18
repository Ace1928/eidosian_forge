from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
def purge_inactive_key(client_obj, master_key, **kwargs):
    if utils.is_null_or_empty(master_key):
        return (False, False, 'Purge inactive master key failed as master key is not present.', {})
    try:
        master_key_resp = client_obj.master_key.get(id=None, name=master_key)
        if utils.is_null_or_empty(master_key_resp):
            return (False, False, f"Master key '{master_key}' cannot be purged as it is not present.", {})
        params = utils.remove_null_args(**kwargs)
        client_obj.master_key.purge_inactive(id=master_key_resp.attrs.get('id'), **params)
        return (True, True, f"Purged inactive master key '{master_key}' successfully.", {})
    except Exception as ex:
        return (False, False, f'Purge inactive master key failed |{ex}', {})