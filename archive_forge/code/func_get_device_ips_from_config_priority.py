from __future__ import absolute_import, division, print_function
import csv
import time
from datetime import datetime
from io import BytesIO, StringIO
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_device_ips_from_config_priority(self):
    """
        Retrieve device IPs based on the configuration.
        Parameters:
            -  self (object): An instance of a class used for interacting with Cisco Cisco Catalyst Center.
        Returns:
            list: A list containing device IPs.
        Description:
            This method retrieves device IPs based on the priority order specified in the configuration.
            It first checks if device IPs are available. If not, it checks hostnames, serial numbers,
            and MAC addresses in order and retrieves IPs based on availability.
            If none of the information is available, an empty list is returned.
        """
    device_ips = self.config[0].get('ip_address_list')
    if device_ips:
        return device_ips
    device_hostnames = self.config[0].get('hostname_list')
    if device_hostnames:
        return self.get_device_ips_from_hostname(device_hostnames)
    device_serial_numbers = self.config[0].get('serial_number_list')
    if device_serial_numbers:
        return self.get_device_ips_from_serial_number(device_serial_numbers)
    device_mac_addresses = self.config[0].get('mac_address_list')
    if device_mac_addresses:
        return self.get_device_ips_from_mac_address(device_mac_addresses)
    return []