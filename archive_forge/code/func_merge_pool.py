from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
def merge_pool(client_obj, pool_name, target, **kwargs):
    if utils.is_null_or_empty(pool_name):
        return (False, False, 'Merge pool failed as pool name is not present.', {}, {})
    if utils.is_null_or_empty(target):
        return (False, False, 'Delete pool failed as target pool name is not present.', {}, {})
    try:
        pool_resp = client_obj.pools.get(id=None, name=pool_name)
        if utils.is_null_or_empty(pool_resp):
            return (False, False, f"Merge pools failed as source pool '{pool_name}' is not present.", {}, {})
        target_pool_resp = client_obj.pools.get(id=None, name=target)
        if utils.is_null_or_empty(target_pool_resp):
            return (False, False, f"Merge pools failed as target pool '{target}' is not present.", {}, {})
        params = utils.remove_null_args(**kwargs)
        resp = client_obj.pools.merge(id=pool_resp.attrs.get('id'), target_pool_id=target_pool_resp.attrs.get('id'), **params)
        if hasattr(resp, 'attrs'):
            resp = resp.attrs
        return (True, True, f"Merged target pool '{target}' to pool '{pool_name}' successfully.", {}, resp)
    except Exception as ex:
        return (False, False, f'Merge pool failed | {ex}', {}, {})