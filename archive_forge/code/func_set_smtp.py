from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def set_smtp(module, blade):
    """Configure SMTP settings"""
    changed = False
    current_smtp = blade.smtp.list_smtp().items[0]
    if module.params['host'] and module.params['host'] != current_smtp.relay_host:
        smtp_settings = Smtp(relay_host=module.params['host'])
        changed = True
        if not module.check_mode:
            try:
                blade.smtp.update_smtp(smtp_settings=smtp_settings)
            except Exception:
                module.fail_json(msg='Configuring SMTP relay host failed')
    elif current_smtp.relay_host and (not module.params['host']):
        smtp_settings = Smtp(relay_host='')
        changed = True
        if not module.check_mode:
            try:
                blade.smtp.update_smtp(smtp_settings=smtp_settings)
            except Exception:
                module.fail_json(msg='Configuring SMTP relay host failed')
    if module.params['domain'] != current_smtp.sender_domain:
        smtp_settings = Smtp(sender_domain=module.params['domain'])
        changed = True
        if not module.check_mode:
            try:
                blade.smtp.update_smtp(smtp_settings=smtp_settings)
            except Exception:
                module.fail_json(msg='Configuring SMTP sender domain failed')
    module.exit_json(changed=changed)