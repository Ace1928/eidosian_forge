from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def global_pool_exists(self, name):
    """
        Check if the Global Pool with the given name exists

        Parameters:
            name (str) - The name of the Global Pool to check for existence

        Returns:
            dict - A dictionary containing information about the Global Pool's existence:
            - 'exists' (bool): True if the Global Pool exists, False otherwise.
            - 'id' (str or None): The ID of the Global Pool if it exists, or None if it doesn't.
            - 'details' (dict or None): Details of the Global Pool if it exists, else None.
        """
    global_pool = {'exists': False, 'details': None, 'id': None}
    response = self.dnac._exec(family='network_settings', function='get_global_pool')
    if not isinstance(response, dict):
        self.log('Failed to retrieve the global pool details - Response is not a dictionary', 'CRITICAL')
        return global_pool
    all_global_pool_details = response.get('response')
    global_pool_details = get_dict_result(all_global_pool_details, 'ipPoolName', name)
    self.log('Global ip pool name: {0}'.format(name), 'DEBUG')
    self.log('Global pool details: {0}'.format(global_pool_details), 'DEBUG')
    if not global_pool_details:
        self.log('Global pool {0} does not exist'.format(name), 'INFO')
        return global_pool
    global_pool.update({'exists': True})
    global_pool.update({'id': global_pool_details.get('id')})
    global_pool['details'] = self.get_global_pool_params(global_pool_details)
    self.log('Formatted global pool details: {0}'.format(global_pool), 'DEBUG')
    return global_pool