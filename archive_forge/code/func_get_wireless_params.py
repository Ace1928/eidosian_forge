from __future__ import absolute_import, division, print_function
import time
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_wireless_params(self):
    """
        Prepares the payload for provisioning of the wireless devices

        Parameters:
          - self: The instance of the class containing the 'config' attribute
                  to be validated.
        Returns:
          The method returns an instance of the class with updated attributes:
          - wireless_params: A list of dictionary containing all the values indicating
                          management IP address of the device, hierarchy
                          of the site, AP Location of the wireless controller and details
                          of the interface
        Example:
          Post creation of the validated input, it fetches the required
          paramters and stores it for further processing and calling the
          parameters in other APIs.
        """
    wireless_params = [{'site': self.validated_config[0].get('site_name_hierarchy'), 'managedAPLocations': self.validated_config[0].get('managed_ap_locations')}]
    for ap_loc in wireless_params[0]['managedAPLocations']:
        if self.get_site_type(site_name_hierarchy=ap_loc) != 'floor':
            self.log('Managed AP Location must be a floor', 'CRITICAL')
            self.module.fail_json(msg='Managed AP Location must be a floor', response=[])
    wireless_params[0]['dynamicInterfaces'] = []
    for interface in self.validated_config[0].get('dynamic_interfaces'):
        interface_dict = {'interfaceIPAddress': interface.get('interface_ip_address'), 'interfaceNetmaskInCIDR': interface.get('interface_netmask_in_c_i_d_r'), 'interfaceGateway': interface.get('interface_gateway'), 'lagOrPortNumber': interface.get('lag_or_port_number'), 'vlanId': interface.get('vlan_id'), 'interfaceName': interface.get('interface_name')}
        wireless_params[0]['dynamicInterfaces'].append(interface_dict)
    response = self.dnac_apply['exec'](family='devices', function='get_network_device_by_ip', params={'management_ip_address': self.validated_config[0]['management_ip_address']})
    self.log("Response collected from 'get_network_device_by_ip' is:{0}".format(str(response)), 'DEBUG')
    wireless_params[0]['deviceName'] = response.get('response')[0].get('hostname')
    self.log('Parameters collected for the provisioning of wireless device:{0}'.format(wireless_params), 'INFO')
    return wireless_params