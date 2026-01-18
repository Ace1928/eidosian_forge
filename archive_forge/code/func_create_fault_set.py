from __future__ import absolute_import, division, print_function
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell import (
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.powerflex_base \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
from ansible.module_utils.basic import AnsibleModule
def create_fault_set(self, fault_set_name, protection_domain_id):
    """
        Create Fault Set
        :param fault_set_name: Name of the fault set
        :type fault_set_name: str
        :param protection_domain_id: ID of the protection domain
        :type protection_domain_id: str
        :return: Boolean indicating if create operation is successful
        """
    try:
        if not self.module.check_mode:
            msg = f'Creating fault set with name: {fault_set_name} on protection domain with id: {protection_domain_id}'
            LOG.info(msg)
            self.powerflex_conn.fault_set.create(name=fault_set_name, protection_domain_id=protection_domain_id)
        return self.get_fault_set(fault_set_name=fault_set_name, protection_domain_id=protection_domain_id)
    except Exception as e:
        error_msg = f'Create fault set {fault_set_name} operation failed with error {str(e)}'
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)