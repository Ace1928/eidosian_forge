from __future__ import absolute_import, division, print_function
import csv
import time
from datetime import datetime
from io import BytesIO, StringIO
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def is_device_exist_for_update(self, device_to_update):
    """
        Check if the device(s) exist in Cisco Catalyst Center for update operation.
        Args:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
            device_to_update (list): A list of device(s) to be be checked present in Cisco Catalyst Center.
        Returns:
            bool: True if at least one of the devices to be updated exists in Cisco Catalyst Center,
                False otherwise.
        Description:
            This function checks if any of the devices specified in the 'device_to_update' list
            exists in Cisco Catalyst Center. It iterates through the list of devices and compares
            each device with the list of devices present in Cisco Catalyst Center obtained from
            'self.have.get("device_in_ccc")'. If a match is found, it sets 'device_exist' to True
            and breaks the loop.
        """
    device_exist = False
    for device in device_to_update:
        if device in self.have.get('device_in_ccc'):
            device_exist = True
            break
    return device_exist