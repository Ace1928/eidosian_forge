from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_have_reserve_pool(self, config):
    """
        Get the current Reserved Pool information from Cisco Catalyst Center
        based on the provided playbook details.
        Check this API using check_return_status

        Parameters:
            config (list of dict) - Playbook details containing Reserved Pool configuration.

        Returns:
            self - The current object with updated information.
        """
    reserve_pool = {'exists': False, 'details': None, 'id': None}
    reserve_pool_details = config.get('reserve_pool_details')
    name = reserve_pool_details.get('name')
    if name is None:
        self.msg = 'Mandatory Parameter name required in reserve_pool_details\n'
        self.status = 'failed'
        return self
    site_name = reserve_pool_details.get('site_name')
    self.log('Site Name: {0}'.format(site_name), 'DEBUG')
    if site_name is None:
        self.msg = "Missing parameter 'site_name' in reserve_pool_details"
        self.status = 'failed'
        return self
    reserve_pool = self.reserve_pool_exists(name, site_name)
    if not reserve_pool.get('success'):
        return self.check_return_status()
    self.log('Reserved pool details: {0}'.format(reserve_pool), 'DEBUG')
    prev_name = reserve_pool_details.get('prev_name')
    if reserve_pool.get('exists') is False and prev_name is not None:
        reserve_pool = self.reserve_pool_exists(prev_name, site_name)
        if not reserve_pool.get('success'):
            return self.check_return_status()
        if reserve_pool.get('exists') is False:
            self.msg = "Prev name {0} doesn't exist in reserve_pool_details".format(prev_name)
            self.status = 'failed'
            return self
    self.log('Reserved pool exists: {0}'.format(reserve_pool.get('exists')), 'DEBUG')
    self.log('Reserved pool: {0}'.format(reserve_pool.get('details')), 'DEBUG')
    if reserve_pool.get('exists'):
        reserve_pool_details = reserve_pool.get('details')
        if reserve_pool_details.get('ipv6AddressSpace') == 'False':
            reserve_pool_details.update({'ipv6AddressSpace': False})
        else:
            reserve_pool_details.update({'ipv6AddressSpace': True})
    self.log('Reserved pool details: {0}'.format(reserve_pool), 'DEBUG')
    self.have.update({'reservePool': reserve_pool})
    self.msg = 'Collecting the reserve pool details from the Cisco Catalyst Center'
    self.status = 'success'
    return self