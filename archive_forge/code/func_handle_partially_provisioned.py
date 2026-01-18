from __future__ import absolute_import, division, print_function
import csv
import time
from datetime import datetime
from io import BytesIO, StringIO
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def handle_partially_provisioned(self, provision_count, device_type):
    """
        Handle partial success in provisioning for devices(Wired/Wireless).
        Parameters:
            - self (object): An instance of a class used for interacting with Cisco Catalyst Center.
            - provision_count (int): The count of devices that were successfully provisioned.
            - device_type (str): The type or category of the provisioned devices(Wired/Wireless).
        Return:
            None
        Description:
            This method updates the status, result, and logs a partial success message indicating that provisioning was successful
            for a certain number of devices(Wired/Wireless).
        """
    self.status = 'success'
    self.result['changed'] = True
    self.log('{0} Devices provisioned successfully partially for {1} devices'.format(device_type, provision_count), 'INFO')