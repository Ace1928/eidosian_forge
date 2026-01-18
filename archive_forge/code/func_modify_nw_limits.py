from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
def modify_nw_limits(self, protection_domain_id, nw_modify_dict, create_flag=False):
    """
        Modify Protection domain attributes
        :param protection_domain_id: ID of the protection domain
        :type protection_domain_id: str
        :param nw_modify_dict: Dictionary containing the attributes of
                               protection domain which are to be updated
        :type nw_modify_dict: dict
        :param create_flag: Flag to indicate whether modify operation is
                            followed by create operation or not
        :type create_flag: bool
        :return: Boolean indicating if the operation is successful
        """
    try:
        msg = 'Dict containing network modify params {0}'.format(str(nw_modify_dict))
        LOG.info(msg)
        if 'rebuild_limit' in nw_modify_dict or 'rebalance_limit' in nw_modify_dict or 'vtree_migration_limit' in nw_modify_dict or ('overall_limit' in nw_modify_dict):
            self.powerflex_conn.protection_domain.network_limits(protection_domain_id=protection_domain_id, rebuild_limit=nw_modify_dict['rebuild_limit'], rebalance_limit=nw_modify_dict['rebalance_limit'], vtree_migration_limit=nw_modify_dict['vtree_migration_limit'], overall_limit=nw_modify_dict['overall_limit'])
            msg = 'The Network limits are updated to {0}, {1}, {2}, {3} successfully.'.format(nw_modify_dict['rebuild_limit'], nw_modify_dict['rebalance_limit'], nw_modify_dict['vtree_migration_limit'], nw_modify_dict['overall_limit'])
            LOG.info(msg)
        return True
    except Exception as e:
        if create_flag:
            err_msg = 'Create protection domain is successful, but failed to update the network limits {0} with error {1}'.format(protection_domain_id, str(e))
        else:
            err_msg = 'Failed to update the network limits of protection domain {0} with error {1}'.format(protection_domain_id, str(e))
        LOG.error(err_msg)
        self.module.fail_json(msg=err_msg)