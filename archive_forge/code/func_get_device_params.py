from __future__ import absolute_import, division, print_function
import csv
import time
from datetime import datetime
from io import BytesIO, StringIO
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_device_params(self, params):
    """
        Extract and store device parameters from the playbook for device processing in Cisco Catalyst Center.
        Parameters:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
            params (dict): A dictionary containing device parameters retrieved from the playbook.
        Returns:
            dict: A dictionary containing the extracted device parameters.
        Description:
            This function will extract and store parameters in dictionary for adding, updating, editing, or deleting devices Cisco Catalyst Center.
        """
    device_param = {'cliTransport': params.get('cli_transport'), 'enablePassword': params.get('enable_password'), 'password': params.get('password'), 'ipAddress': params.get('ip_address_list'), 'snmpAuthPassphrase': params.get('snmp_auth_passphrase'), 'snmpAuthProtocol': params.get('snmp_auth_protocol'), 'snmpMode': params.get('snmp_mode'), 'snmpPrivPassphrase': params.get('snmp_priv_passphrase'), 'snmpPrivProtocol': params.get('snmp_priv_protocol'), 'snmpROCommunity': params.get('snmp_ro_community'), 'snmpRWCommunity': params.get('snmp_rw_community'), 'snmpRetry': params.get('snmp_retry'), 'snmpTimeout': params.get('snmp_timeout'), 'snmpUserName': params.get('snmp_username'), 'userName': params.get('username'), 'computeDevice': params.get('compute_device'), 'extendedDiscoveryInfo': params.get('extended_discovery_info'), 'httpPassword': params.get('http_password'), 'httpPort': params.get('http_port'), 'httpSecure': params.get('http_secure'), 'httpUserName': params.get('http_username'), 'netconfPort': params.get('netconf_port'), 'serialNumber': params.get('serial_number'), 'snmpVersion': params.get('snmp_version'), 'type': params.get('type'), 'updateMgmtIPaddressList': params.get('update_mgmt_ipaddresslist'), 'forceSync': params.get('force_sync'), 'cleanConfig': params.get('clean_config')}
    if device_param.get('updateMgmtIPaddressList'):
        device_mngmt_dict = device_param.get('updateMgmtIPaddressList')[0]
        device_param['updateMgmtIPaddressList'][0] = {}
        device_param['updateMgmtIPaddressList'][0].update({'existMgmtIpAddress': device_mngmt_dict.get('exist_mgmt_ipaddress'), 'newMgmtIpAddress': device_mngmt_dict.get('new_mgmt_ipaddress')})
    return device_param