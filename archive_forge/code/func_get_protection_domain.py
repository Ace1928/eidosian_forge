from __future__ import (absolute_import, division, print_function)
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
def get_protection_domain(self, protection_domain_name=None, protection_domain_id=None):
    """
        Get protection domain details
        :param protection_domain_name: Name of the protection domain
        :param protection_domain_id: ID of the protection domain
        :return: Protection domain details if exists
        :rtype: dict
        """
    name_or_id = protection_domain_id if protection_domain_id else protection_domain_name
    try:
        if protection_domain_id:
            pd_details = self.powerflex_conn.protection_domain.get(filter_fields={'id': protection_domain_id})
        else:
            pd_details = self.powerflex_conn.protection_domain.get(filter_fields={'name': protection_domain_name})
        if len(pd_details) == 0:
            error_msg = "Unable to find the protection domain with '%s'." % name_or_id
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)
        return pd_details[0]
    except Exception as e:
        error_msg = "Failed to get the protection domain '%s' with error '%s'" % (name_or_id, str(e))
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)