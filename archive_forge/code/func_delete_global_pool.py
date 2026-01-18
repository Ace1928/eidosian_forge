from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def delete_global_pool(self, name):
    """
        Delete a Global Pool by name in Cisco Catalyst Center

        Parameters:
            name (str) - The name of the Global Pool to be deleted.

        Returns:
            self
        """
    global_pool_exists = self.have.get('globalPool').get('exists')
    result_global_pool = self.result.get('response')[0].get('globalPool')
    if not global_pool_exists:
        result_global_pool.get('response').update({name: 'Global Pool not found'})
        self.msg = 'Global pool Not Found'
        self.status = 'success'
        return self
    response = self.dnac._exec(family='network_settings', function='delete_global_ip_pool', params={'id': self.have.get('globalPool').get('id')})
    self.check_execution_response_status(response).check_return_status()
    executionid = response.get('executionId')
    result_global_pool = self.result.get('response')[0].get('globalPool')
    result_global_pool.get('response').update({name: {}})
    result_global_pool.get('response').get(name).update({'Execution Id': executionid})
    result_global_pool.get('msg').update({name: 'Pool deleted successfully'})
    self.msg = 'Global pool - {0} deleted successfully'.format(name)
    self.status = 'success'
    return self