from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
def create_protection_domain(self, protection_domain_name):
    """
        Create Protection Domain
        :param protection_domain_name: Name of the protection domain
        :type protection_domain_name: str
        :return: Boolean indicating if create operation is successful
        """
    try:
        LOG.info('Creating protection domain with name: %s ', protection_domain_name)
        self.powerflex_conn.protection_domain.create(name=protection_domain_name)
        return True
    except Exception as e:
        error_msg = "Create protection domain '%s' operation failed with error '%s'" % (protection_domain_name, str(e))
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)