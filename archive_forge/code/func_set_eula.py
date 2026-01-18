from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def set_eula(module, blade):
    """Sign EULA"""
    changed = False
    if not module.check_mode:
        current_eula = list(blade.get_arrays_eula().items)[0].signature
        if not current_eula.accepted:
            if current_eula.company != module.params['company'] or current_eula.title != module.params['title'] or current_eula.name != module.params['name']:
                signature = EulaSignature(company=module.params['company'], title=module.params['title'], name=module.params['name'])
                eula_body = Eula(signature=signature)
                if not module.check_mode:
                    changed = True
                    rc = blade.patch_arrays_eula(eula=eula_body)
                    if rc.status_code != 200:
                        module.fail_json(msg='Signing EULA failed')
    module.exit_json(changed=changed)