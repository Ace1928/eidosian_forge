from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
def is_id_or_new_name_in_create(self):
    """Checking if protection domain id or new names present in create """
    if self.module.params['protection_domain_new_name'] or self.module.params['protection_domain_id']:
        error_msg = 'protection_domain_new_name/protection_domain_id are not supported during creation of protection domain. Please try with protection_domain_name.'
        LOG.info(error_msg)
        self.module.fail_json(msg=error_msg)