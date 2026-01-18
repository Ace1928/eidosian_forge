from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_have_global_pool(self, config):
    """
        Get the current Global Pool information from
        Cisco Catalyst Center based on the provided playbook details.
        check this API using check_return_status.

        Parameters:
            config (dict) - Playbook details containing Global Pool configuration.

        Returns:
            self - The current object with updated information.
        """
    global_pool = {'exists': False, 'details': None, 'id': None}
    global_pool_settings = config.get('global_pool_details').get('settings')
    if global_pool_settings is None:
        self.msg = 'settings in global_pool_details is missing in the playbook'
        self.status = 'failed'
        return self
    global_pool_ippool = global_pool_settings.get('ip_pool')
    if global_pool_ippool is None:
        self.msg = 'ip_pool in global_pool_details is missing in the playbook'
        self.status = 'failed'
        return self
    name = global_pool_ippool[0].get('name')
    if name is None:
        self.msg = 'Mandatory Parameter name required'
        self.status = 'failed'
        return self
    global_pool = self.global_pool_exists(name)
    self.log('Global pool details: {0}'.format(global_pool), 'DEBUG')
    prev_name = global_pool_ippool[0].get('prev_name')
    if global_pool.get('exists') is False and prev_name is not None:
        global_pool = self.global_pool_exists(prev_name)
        if global_pool.get('exists') is False:
            self.msg = "Prev name {0} doesn't exist in global_pool_details".format(prev_name)
            self.status = 'failed'
            return self
    self.log('Global pool exists: {0}'.format(global_pool.get('exists')), 'DEBUG')
    self.log('Current Site: {0}'.format(global_pool.get('details')), 'DEBUG')
    self.have.update({'globalPool': global_pool})
    self.msg = 'Collecting the global pool details from the Cisco Catalyst Center'
    self.status = 'success'
    return self