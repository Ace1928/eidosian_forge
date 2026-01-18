from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
def create_chap_user(client_obj, user_name, password, **kwargs):
    if utils.is_null_or_empty(user_name):
        return (False, False, 'Create chap user failed as user is not present.', {}, {})
    if utils.is_null_or_empty(password):
        return (False, False, 'Create chap user failed as password is not present.', {}, {})
    try:
        user_resp = client_obj.chap_users.get(id=None, name=user_name)
        if utils.is_null_or_empty(user_resp):
            params = utils.remove_null_args(**kwargs)
            user_resp = client_obj.chap_users.create(name=user_name, password=password, **params)
            return (True, True, f"Chap user '{user_name}' created successfully.", {}, user_resp.attrs)
        else:
            return (False, False, f"Chap user '{user_name}' cannot be created as it is already present in given state.", {}, user_resp.attrs)
    except Exception as ex:
        return (False, False, f'Chap user creation failed |{ex}', {}, {})