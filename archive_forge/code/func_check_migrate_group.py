from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
def check_migrate_group(client_obj, group_name):
    if utils.is_null_or_empty(group_name):
        return (False, False, 'Check migrate for group failed as it is not present.', {})
    try:
        group_resp = client_obj.groups.get(id=None, name=group_name)
        if utils.is_null_or_empty(group_resp):
            return (False, False, f"Check migrate for group '{group_name}' cannot be done as it is not present.", {})
        client_obj.groups.check_migrate(id=group_resp.attrs.get('id'))
        return (True, True, f"Check migrate for group '{group_name}' done successfully.", {})
    except Exception as ex:
        return (False, False, f"Check migrate for group failed | '{ex}'", {})