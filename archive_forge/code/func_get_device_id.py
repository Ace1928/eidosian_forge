from __future__ import absolute_import, division, print_function
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
from ansible.module_utils.basic import AnsibleModule
import os
import time
def get_device_id(self, params):
    """
        Retrieve the unique device ID based on the provided parameters.
        Parameters:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
            params (dict): A dictionary containing parameters to filter devices.
        Returns:
            str: The unique device ID corresponding to the filtered device.
        Description:
            This function sends a request to Cisco Catalyst Center to retrieve a list of devices based on the provided
            filtering parameters. If a single matching device is found, it extracts and returns the device ID. If
            no device or multiple devices match the criteria, it raises an exception.
        """
    device_id = None
    response = self.dnac._exec(family='devices', function='get_device_list', params=params)
    self.log("Received API response from 'get_device_list': {0}".format(str(response)), 'DEBUG')
    device_list = response.get('response')
    if len(device_list) == 1:
        device_id = device_list[0].get('id')
        self.log('Device Id: {0}'.format(str(device_id)), 'INFO')
    else:
        self.msg = "Device with params: '{0}' not found in Cisco Catalyst Center so can't fetch the device id".format(str(params))
        self.log(self.msg, 'WARNING')
    return device_id