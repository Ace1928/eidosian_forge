from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
def create_prot_schedule(client_obj, prot_schedule_name, **kwargs):
    if utils.is_null_or_empty(prot_schedule_name):
        return (False, False, 'Create protection schedule failed as protection schedule name is not present.', {}, {})
    try:
        prot_schedule_resp = client_obj.protection_schedules.get(id=None, name=prot_schedule_name, volcoll_or_prottmpl_type=kwargs['volcoll_or_prottmpl_type'], volcoll_or_prottmpl_id=kwargs['volcoll_or_prottmpl_id'])
        if utils.is_null_or_empty(prot_schedule_resp):
            params = utils.remove_null_args(**kwargs)
            prot_schedule_resp = client_obj.protection_schedules.create(name=prot_schedule_name, **params)
            return (True, True, f"Created protection schedule '{prot_schedule_name}' successfully.", {}, prot_schedule_resp.attrs)
        else:
            return (False, False, f"Cannot create protection schedule '{prot_schedule_name}' as it is already present in given state.", {}, prot_schedule_resp.attrs)
    except Exception as ex:
        return (False, False, f'Protection schedule creation failed | {ex}', {}, {})