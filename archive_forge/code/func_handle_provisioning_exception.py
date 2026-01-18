from __future__ import absolute_import, division, print_function
import csv
import time
from datetime import datetime
from io import BytesIO, StringIO
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def handle_provisioning_exception(self, device_ip, exception, device_type):
    """
        Handle an exception during the provisioning process of Wired/Wireless device..
        Parameters:
            - self (object): An instance of a class used for interacting with Cisco Catalyst Center.
            - device_ip (str): The IP address of the device involved in provisioning.
            - exception (Exception): The exception raised during provisioning.
            - device_type (str): The type or category of the provisioned device(Wired/Wireless).
        Return:
            None
        Description:
            This method logs an error message indicating an exception occurred during the provisioning process for a device.
        """
    error_message = 'Error while Provisioning the {0} device {1} in Cisco Catalyst Center: {2}'.format(device_type, device_ip, str(exception))
    self.log(error_message, 'ERROR')