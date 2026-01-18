from __future__ import absolute_import, division, print_function
import csv
import time
from datetime import datetime
from io import BytesIO, StringIO
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def handle_all_already_provisioned(self, device_ips, device_type):
    """
        Handle successful provisioning for all devices(Wired/Wireless).
        Parameters:
            - self (object): An instance of a class used for interacting with Cisco Catalyst Center.
            - device_type (str): The type or category of the provisioned device(Wired/Wireless).
        Return:
            None
        Description:
            This method updates the status, result, and logs the successful provisioning for all devices(Wired/Wireless).
        """
    self.status = 'success'
    self.msg = "All the {0} Devices '{1}' given in the playbook are already Provisioned".format(device_type, str(device_ips))
    self.log(self.msg, 'INFO')
    self.result['response'] = self.msg
    self.result['changed'] = False