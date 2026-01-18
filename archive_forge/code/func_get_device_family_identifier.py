from __future__ import absolute_import, division, print_function
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
from ansible.module_utils.basic import AnsibleModule
import os
import time
def get_device_family_identifier(self, family_name):
    """
        Retrieve and store the device family identifier based on the provided family name.
        Parameters:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
            family_name (str): The name of the device family for which to retrieve the identifier.
        Returns:
            None
        Raises:
            AnsibleFailJson: If the family name is not found in the response.
        Description:
            This function sends a request to Cisco Catalyst Center to retrieve a list of device family identifiers.It then
            searches for a specific family name within the response and stores its associated identifier. If the family
            name is found, the identifier is stored; otherwise, an exception is raised.
        """
    have = {}
    response = self.dnac._exec(family='software_image_management_swim', function='get_device_family_identifiers')
    self.log("Received API response from 'get_device_family_identifiers': {0}".format(str(response)), 'DEBUG')
    device_family_db = response.get('response')
    if device_family_db:
        device_family_details = get_dict_result(device_family_db, 'deviceFamily', family_name)
        if device_family_details:
            device_family_identifier = device_family_details.get('deviceFamilyIdentifier')
            have['device_family_identifier'] = device_family_identifier
            self.log('Family device indentifier: {0}'.format(str(device_family_identifier)), 'INFO')
        else:
            self.msg = 'Device Family: {0} not found'.format(str(family_name))
            self.log(self.msg, 'ERROR')
            self.module.fail_json(msg=self.msg, response=[self.msg])
        self.have.update(have)