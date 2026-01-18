from __future__ import absolute_import, division, print_function
import csv
import time
from datetime import datetime
from io import BytesIO, StringIO
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def check_credential_update(self):
    """
        Checks if the credentials for devices in the configuration match the updated values in Cisco Catalyst Center.
        Parameters:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
        Returns:
            bool: True if the credentials match the updated values, False otherwise.
        Description:
            This method triggers the export API in Cisco Catalyst Center to obtain the updated credential details for
            the specified devices. It then decrypts and reads the CSV file containing the updated credentials,
            comparing them with the credentials specified in the configuration.
        """
    device_ips = self.get_device_ips_from_config_priority()
    device_uuids = self.get_device_ids(device_ips)
    password = 'Testing@123'
    payload_params = {'deviceUuids': device_uuids, 'password': password, 'operationEnum': '0'}
    response = self.trigger_export_api(payload_params)
    self.check_return_status()
    csv_reader = self.decrypt_and_read_csv(response, password)
    self.check_return_status()
    device_data = next(csv_reader, None)
    if not device_data:
        return False
    csv_data_dict = {'snmp_retry': device_data['snmp_retries'], 'username': device_data['cli_username'], 'password': device_data['cli_password'], 'enable_password': device_data['cli_enable_password'], 'snmp_username': device_data['snmpv3_user_name'], 'snmp_auth_protocol': device_data['snmpv3_auth_type']}
    config = self.config[0]
    for key in csv_data_dict:
        if key in config and csv_data_dict[key] is not None:
            if key == 'snmp_retry' and int(csv_data_dict[key]) != int(config[key]):
                return False
            elif csv_data_dict[key] != config[key]:
                return False
    return True