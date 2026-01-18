from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
def modify_rf_limits(self, protection_domain_id, rf_modify_dict, create_flag):
    """
        Modify Protection domain attributes
        :param protection_domain_id: ID of the protection domain
        :type protection_domain_id: str
        :param rf_modify_dict: Dict containing the attributes of rf cache
                               which are to be updated
        :type rf_modify_dict: dict
        :param create_flag: Flag to indicate whether modify operation is
                            followed by create operation or not
        :type create_flag: bool
        :return: Boolean indicating if the operation is successful
        """
    try:
        msg = 'Dict containing network modify params {0}'.format(str(rf_modify_dict))
        LOG.info(msg)
        if 'is_enabled' in rf_modify_dict and rf_modify_dict['is_enabled'] is not None:
            self.powerflex_conn.protection_domain.set_rfcache_enabled(protection_domain_id, rf_modify_dict['is_enabled'])
            msg = "The RFcache is enabled to '%s' successfully." % rf_modify_dict['is_enabled']
            LOG.info(msg)
        if 'page_size' in rf_modify_dict or 'max_io_limit' in rf_modify_dict or 'pass_through_mode' in rf_modify_dict:
            self.powerflex_conn.protection_domain.rfcache_parameters(protection_domain_id=protection_domain_id, page_size=rf_modify_dict['page_size'], max_io_limit=rf_modify_dict['max_io_limit'], pass_through_mode=rf_modify_dict['pass_through_mode'])
            msg = "The RFcache parameters are updated to {0}, {1},{2}.'".format(rf_modify_dict['page_size'], rf_modify_dict['max_io_limit'], rf_modify_dict['pass_through_mode'])
            LOG.info(msg)
        return True
    except Exception as e:
        if create_flag:
            err_msg = 'Create protection domain is successful, but failed to update the rf cache limits {0} with error {1}'.format(protection_domain_id, str(e))
        else:
            err_msg = 'Failed to update the rf cache limits of protection domain {0} with error {1}'.format(protection_domain_id, str(e))
        LOG.error(err_msg)
        self.module.fail_json(msg=err_msg)