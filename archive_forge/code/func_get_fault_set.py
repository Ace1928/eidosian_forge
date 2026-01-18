from __future__ import (absolute_import, division, print_function)
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
def get_fault_set(self, fault_set_name=None, fault_set_id=None, protection_domain_id=None):
    """Get fault set details
            :param fault_set_name: Name of the fault set
            :param fault_set_id: Id of the fault set
            :param protection_domain_id: ID of the protection domain
            :return: Fault set details
            :rtype: dict
        """
    name_or_id = fault_set_id if fault_set_id else fault_set_name
    try:
        fs_details = {}
        if fault_set_id:
            fs_details = self.powerflex_conn.fault_set.get(filter_fields={'id': name_or_id})
        if fault_set_name:
            fs_details = self.powerflex_conn.fault_set.get(filter_fields={'name': name_or_id, 'protectionDomainId': protection_domain_id})
        if not fs_details:
            msg = f'Unable to find the fault set with {name_or_id}'
            LOG.info(msg)
            return None
        return fs_details[0]
    except Exception as e:
        error_msg = f"Failed to get the fault set '{name_or_id}' with error '{str(e)}'"
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)