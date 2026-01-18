from __future__ import absolute_import, division, print_function
import csv
import time
from datetime import datetime
from io import BytesIO, StringIO
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def handle_failed_provisioning(self, device_ip, execution_details, device_type):
    """
        Handle failed provisioning of Wired/Wireless device.
        Parameters:
            - self (object): An instance of a class used for interacting with Cisco Catalyst Center.
            - device_ip (str): The IP address of the device that failed provisioning.
            - execution_details (dict): Details of the failed provisioning execution in key "failureReason" indicating reason for failure.
            - device_type (str): The type or category of the provisioned device(Wired/Wireless).
        Return:
            None
        Description:
            This method updates the status, result, and logs the failure of provisioning for a device.
        """
    self.status = 'failed'
    failure_reason = execution_details.get('failureReason', 'Unknown failure reason')
    self.msg = '{0} Device Provisioning failed for {1} because of {2}'.format(device_type, device_ip, failure_reason)
    self.log(self.msg, 'WARNING')