from __future__ import absolute_import, division, print_function
import csv
import time
from datetime import datetime
from io import BytesIO, StringIO
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def handle_successful_provisioning(self, device_ip, execution_details, device_type):
    """
        Handle successful provisioning of Wired/Wireless device.
        Parameters:
            - self (object): An instance of a class used for interacting with Cisco Catalyst Center.
            - device_ip (str): The IP address of the provisioned device.
            - execution_details (str): Details of the provisioning execution.
            - device_type (str): The type or category of the provisioned device(Wired/Wireless).
        Return:
            None
        Description:
            This method updates the status, result, and logs the successful provisioning of a device.
        """
    self.status = 'success'
    self.result['changed'] = True
    self.result['response'] = execution_details
    self.log('{0} Device {1} provisioned successfully!!'.format(device_type, device_ip), 'INFO')