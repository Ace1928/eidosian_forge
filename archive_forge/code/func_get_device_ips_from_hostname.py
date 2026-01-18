from __future__ import absolute_import, division, print_function
import csv
import time
from datetime import datetime
from io import BytesIO, StringIO
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_device_ips_from_hostname(self, hostname_list):
    """
        Get the list of unique device IPs for list of specified hostnames of devices in Cisco Catalyst Center.
        Parameters:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
            hostname_list (list): The hostnames of devices for which you want to retrieve the device IPs.
        Returns:
            list: The list of unique device IPs for the specified devices hostname list.
        Description:
            Queries Cisco Catalyst Center to retrieve the unique device IP's associated with a device having the specified
            list of hostnames. If a device is not found in Cisco Catalyst Center, an error log message is printed.
        """
    device_ips = []
    for hostname in hostname_list:
        try:
            response = self.dnac._exec(family='devices', function='get_device_list', params={'hostname': hostname})
            if response:
                self.log("Received API response from 'get_device_list': {0}".format(str(response)), 'DEBUG')
                response = response.get('response')
                if response:
                    device_ip = response[0]['managementIpAddress']
                    if device_ip:
                        device_ips.append(device_ip)
        except Exception as e:
            error_message = 'Exception occurred while fetching device from Cisco Catalyst Center: {0}'.format(str(e))
            self.log(error_message, 'ERROR')
    return device_ips