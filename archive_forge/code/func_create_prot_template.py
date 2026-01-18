from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
def create_prot_template(client_obj, prot_template_name, **kwargs):
    if utils.is_null_or_empty(prot_template_name):
        return (False, False, 'Create protection template failed as protection template name is not present.', {}, {})
    try:
        prot_template_resp = client_obj.protection_templates.get(id=None, name=prot_template_name)
        if utils.is_null_or_empty(prot_template_resp):
            params = utils.remove_null_args(**kwargs)
            prot_template_resp = client_obj.protection_templates.create(name=prot_template_name, **params)
            return (True, True, f"Protection template '{prot_template_name}' created successfully.", {}, prot_template_resp.attrs)
        else:
            return (False, False, f"Protection template '{prot_template_name}' cannot be created as it is already present in given state.", {}, prot_template_resp.attrs)
    except Exception as ex:
        return (False, False, f'Protection template creation failed | {ex}', {}, {})