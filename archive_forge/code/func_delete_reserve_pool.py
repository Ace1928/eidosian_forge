from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def delete_reserve_pool(self, name):
    """
        Delete a Reserve Pool by name in Cisco Catalyst Center

        Parameters:
            name (str) - The name of the Reserve Pool to be deleted.

        Returns:
            self
        """
    reserve_pool_exists = self.have.get('reservePool').get('exists')
    result_reserve_pool = self.result.get('response')[1].get('reservePool')
    if not reserve_pool_exists:
        result_reserve_pool.get('response').update({name: 'Reserve Pool not found'})
        self.msg = 'Reserved Ip Subpool Not Found'
        self.status = 'success'
        return self
    self.log('Reserved IP pool scheduled for deletion: {0}'.format(self.have.get('reservePool').get('name')), 'INFO')
    _id = self.have.get('reservePool').get('id')
    self.log('Reserved pool {0} id: {1}'.format(name, _id), 'DEBUG')
    response = self.dnac._exec(family='network_settings', function='release_reserve_ip_subpool', params={'id': _id})
    self.check_execution_response_status(response).check_return_status()
    executionid = response.get('executionId')
    result_reserve_pool = self.result.get('response')[1].get('reservePool')
    result_reserve_pool.get('response').update({name: {}})
    result_reserve_pool.get('response').get(name).update({'Execution Id': executionid})
    result_reserve_pool.get('msg').update({name: 'Ip subpool reservation released successfully'})
    self.msg = 'Reserved pool - {0} released successfully'.format(name)
    self.status = 'success'
    return self