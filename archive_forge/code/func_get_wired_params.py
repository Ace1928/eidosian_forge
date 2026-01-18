from __future__ import absolute_import, division, print_function
import time
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_wired_params(self):
    """
        Prepares the payload for provisioning of the wired devices

        Parameters:
          - self: The instance of the class containing the 'config' attribute
                  to be validated.
        Returns:
          The method returns an instance of the class with updated attributes:
          - wired_params: A dictionary containing all the values indicating
                          management IP address of the device and the hierarchy
                          of the site.
        Example:
          Post creation of the validated input, it fetches the required
          paramters and stores it for further processing and calling the
          parameters in other APIs.
        """
    wired_params = {'deviceManagementIpAddress': self.validated_config[0]['management_ip_address'], 'siteNameHierarchy': self.validated_config[0].get('site_name_hierarchy')}
    self.log('Parameters collected for the provisioning of wired device:{0}'.format(wired_params), 'INFO')
    return wired_params