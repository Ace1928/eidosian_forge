from __future__ import absolute_import, division, print_function
import csv
import time
from datetime import datetime
from io import BytesIO, StringIO
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def handle_all_failed_provision(self, device_type):
    """
        Handle failure of provisioning for all devices(Wired/Wireless).
        Parameters:
            - self (object): An instance of a class used for interacting with Cisco Catalyst Center.
            - device_type (str): The type or category of the devices(Wired/Wireless).
        Return:
            None
        Description:
            This method updates the status and logs a failure message indicating that
            provisioning failed for all devices of a specific type.
        """
    self.status = 'failed'
    self.msg = '{0} Device Provisioning failed for all devices'.format(device_type)
    self.log(self.msg, 'INFO')