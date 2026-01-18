from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
def modify_pd_attributes(self, protection_domain_id, modify_dict, create_flag=False):
    """
        Modify Protection domain attributes
        :param protection_domain_id: ID of the protection domain
        :type protection_domain_id: str
        :param modify_dict: Dictionary containing the attributes of
                            protection domain which are to be updated
        :type modify_dict: dict
        :param create_flag: Flag to indicate whether modify operation is
                            followed by create operation or not
        :type create_flag: bool
        :return: Boolean indicating if the operation is successful
        """
    try:
        msg = "Dictionary containing attributes which need to be updated are '%s'." % str(modify_dict)
        LOG.info(msg)
        if 'protection_domain_new_name' in modify_dict:
            self.powerflex_conn.protection_domain.rename(protection_domain_id, modify_dict['protection_domain_new_name'])
            msg = "The name of the protection domain is updated to '%s' successfully." % modify_dict['protection_domain_new_name']
            LOG.info(msg)
        if 'is_active' in modify_dict and modify_dict['is_active']:
            self.powerflex_conn.protection_domain.activate(protection_domain_id, modify_dict['is_active'])
            msg = "The protection domain is activated successfully, by setting as is_active: '%s' " % modify_dict['is_active']
            LOG.info(msg)
        if 'is_active' in modify_dict and (not modify_dict['is_active']):
            self.powerflex_conn.protection_domain.inactivate(protection_domain_id, modify_dict['is_active'])
            msg = "The protection domain is inactivated successfully, by setting as is_active: '%s' " % modify_dict['is_active']
            LOG.info(msg)
        return True
    except Exception as e:
        if create_flag:
            err_msg = 'Create protection domain is successful, but failed to update the protection domain {0} with error {1}'.format(protection_domain_id, str(e))
        else:
            err_msg = 'Failed to update the protection domain {0} with error {1}'.format(protection_domain_id, str(e))
        LOG.error(err_msg)
        self.module.fail_json(msg=err_msg)