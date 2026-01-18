from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
def create_igroup(client_obj, initiator_group_name, **kwargs):
    if utils.is_null_or_empty(initiator_group_name):
        return (False, False, 'Initiator group creation failed. Initiator group name is null.', {}, {})
    try:
        ig_resp = client_obj.initiator_groups.get(id=None, name=initiator_group_name)
        if utils.is_null_or_empty(ig_resp):
            params = utils.remove_null_args(**kwargs)
            ig_resp = client_obj.initiator_groups.create(name=initiator_group_name, **params)
            return (True, True, f"Created initiator Group '{initiator_group_name}' successfully.", {}, ig_resp.attrs)
        else:
            return (False, False, f"Cannot create initiator Group '{initiator_group_name}' as it is already present in given state.", {}, {})
    except Exception as ex:
        return (False, False, f'Initiator group creation failed | {ex}', {}, {})