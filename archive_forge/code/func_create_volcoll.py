from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
def create_volcoll(client_obj, volcoll_name, **kwargs):
    if utils.is_null_or_empty(volcoll_name):
        return (False, False, 'Create volume collection failed as volume collection is not present.', {}, {})
    try:
        volcoll_resp = client_obj.volume_collections.get(id=None, name=volcoll_name)
        if utils.is_null_or_empty(volcoll_resp):
            params = utils.remove_null_args(**kwargs)
            volcoll_resp = client_obj.volume_collections.create(name=volcoll_name, **params)
            return (True, True, f"Created volume collection '{volcoll_name}' successfully.", {}, volcoll_resp.attrs)
        else:
            return (False, False, f"Volume collection '{volcoll_name}' cannot be created as it is already present in given state.", {}, {})
    except Exception as ex:
        return (False, False, f'Volume collection creation failed | {ex}', {}, {})