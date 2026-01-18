from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
def delete_chap_user(client_obj, user_name):
    if utils.is_null_or_empty(user_name):
        return (False, False, 'Delete chap user failed as user is not present.', {})
    try:
        user_resp = client_obj.chap_users.get(id=None, name=user_name)
        if utils.is_null_or_empty(user_resp):
            return (False, False, f"Chap user '{user_name}' cannot be deleted as it is not present.", {})
        client_obj.chap_users.delete(id=user_resp.attrs.get('id'))
        return (True, True, f"Deleted chap user '{user_name}' successfully.", {})
    except Exception as ex:
        return (False, False, f'Delete chap user failed |{ex}', {})