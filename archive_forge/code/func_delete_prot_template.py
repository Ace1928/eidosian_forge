from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
def delete_prot_template(client_obj, prot_template_name):
    if utils.is_null_or_empty(prot_template_name):
        return (False, False, 'Protection template deletion failed as protection template name is not present.', {})
    try:
        prot_template_resp = client_obj.protection_templates.get(id=None, name=prot_template_name)
        if utils.is_null_or_empty(prot_template_resp):
            return (False, False, f"Protection template '{prot_template_name}' not present to delete.", {})
        else:
            client_obj.protection_templates.delete(id=prot_template_resp.attrs.get('id'))
            return (True, True, f"Deleted protection template '{prot_template_name}' successfully.", {})
    except Exception as ex:
        return (False, False, f'Protection template deletion failed | {ex}', {})