from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_pnp_params(self, params):
    """
        Store pnp parameters from the playbook for pnp processing in Cisco Catalyst Center.

        Parameters:
          - self: The instance of the class containing the 'config'
                  attribute to be validated.
          - params: The validated params passed from the playbook.
        Returns:
          The method returns an instance of the class with updated attributes:
          - pnp_params: A dictionary containing all the values indicating
                        the type of the site (area/building/floor).
        Example:
          Post creation of the validated input, it fetches the required paramters
          and stores it for further processing and calling the parameters in
          other APIs.
        """
    params_list = params['device_info']
    device_info_list = []
    for param in params_list:
        device_dict = {}
        param['serialNumber'] = param.pop('serial_number')
        if 'is_sudi_required' in param:
            param['isSudiRequired'] = param.pop('is_sudi_required')
        device_dict['deviceInfo'] = param
        device_info_list.append(device_dict)
    self.log('PnP paramters passed are {0}'.format(str(params_list)), 'INFO')
    return device_info_list