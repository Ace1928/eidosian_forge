from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.powerflex_base \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
import copy
def create_sds(self, protection_domain_id, sds_ip_list, sds_ip_state, sds_name, sds_id, sds_new_name, rmcache_enabled=None, rmcache_size=None, fault_set_id=None):
    """Create SDS
            :param protection_domain_id: ID of the Protection Domain
            :type protection_domain_id: str
            :param sds_ip_list: List of one or more IP addresses associated
                                with the SDS over which the data will be
                                transferred.
            :type sds_ip_list: list[dict]
            :param sds_ip_state: SDS IP state
            :type sds_ip_state: str
            :param sds_name: SDS name
            :type sds_name: str
            :param rmcache_enabled: Whether to enable the Read RAM cache
            :type rmcache_enabled: bool
            :param rmcache_size: Read RAM cache size (in MB)
            :type rmcache_size: int
            :param fault_set_id: ID of the Fault Set
            :type fault_set_id: str
            :return: Boolean indicating if create operation is successful
        """
    try:
        self.validate_create(protection_domain_id=protection_domain_id, sds_ip_list=sds_ip_list, sds_ip_state=sds_ip_state, sds_name=sds_name, sds_id=sds_id, sds_new_name=sds_new_name, rmcache_enabled=rmcache_enabled, rmcache_size=rmcache_size, fault_set_id=fault_set_id)
        self.validate_ip_parameter(sds_ip_list)
        if not self.module.check_mode:
            if sds_ip_list and sds_ip_state == 'present-in-sds':
                sds_ip_list = self.restructure_ip_role_dict(sds_ip_list)
            if rmcache_size is not None:
                self.validate_rmcache_size_parameter(rmcache_enabled=rmcache_enabled, rmcache_size=rmcache_size)
                rmcache_size = rmcache_size * 1024
            create_params = 'protection_domain_id: %s, sds_ip_list: %s, sds_name: %s, rmcache_enabled: %s,  rmcache_size_KB: %s,  fault_set_id: %s' % (protection_domain_id, sds_ip_list, sds_name, rmcache_enabled, rmcache_size, fault_set_id)
            LOG.info('Creating SDS with params: %s', create_params)
            self.powerflex_conn.sds.create(protection_domain_id=protection_domain_id, sds_ips=sds_ip_list, name=sds_name, rmcache_enabled=rmcache_enabled, rmcache_size_in_kb=rmcache_size, fault_set_id=fault_set_id)
        return self.get_sds_details(sds_name=sds_name)
    except Exception as e:
        error_msg = f'Create SDS {sds_name} operation failed with error {str(e)}'
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)