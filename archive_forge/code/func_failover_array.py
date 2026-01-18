from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
def failover_array(client_obj, array_name, **kwargs):
    if utils.is_null_or_empty(array_name):
        return (False, False, 'Failover array failed as array name is not present.', {})
    try:
        array_resp = client_obj.arrays.get(id=None, name=array_name)
        if utils.is_null_or_empty(array_resp):
            return (False, False, f"Array '{array_name}' cannot failover as it is not present.", {})
        else:
            params = utils.remove_null_args(**kwargs)
            array_resp = client_obj.arrays.failover(id=array_resp.attrs.get('id'), **params)
            return (True, True, f"Failover array '{array_name}' successfully.", {})
    except Exception as ex:
        return (False, False, f'Array failover failed |{ex}', {})