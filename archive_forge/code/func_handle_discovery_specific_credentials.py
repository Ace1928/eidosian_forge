from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
import time
import re
def handle_discovery_specific_credentials(self, new_object_params=None):
    """
        Method to convert values for create_params API when discovery specific paramters
        are passed as input.

        Parameters:
            - new_object_params: The dictionary storing various parameters for calling the
                                 start discovery API

        Returns:
            - new_object_params: The dictionary storing various parameters for calling the
                                 start discovery API in an updated fashion
        """
    discovery_specific_credentials = self.validated_config[0].get('discovery_specific_credentials')
    cli_credentials_list = discovery_specific_credentials.get('cli_credentials_list')
    http_read_credential = discovery_specific_credentials.get('http_read_credential')
    http_write_credential = discovery_specific_credentials.get('http_write_credential')
    snmp_v2_read_credential = discovery_specific_credentials.get('snmp_v2_read_credential')
    snmp_v2_write_credential = discovery_specific_credentials.get('snmp_v2_write_credential')
    snmp_v3_credential = discovery_specific_credentials.get('snmp_v3_credential')
    net_conf_port = discovery_specific_credentials.get('net_conf_port')
    if cli_credentials_list:
        if not isinstance(cli_credentials_list, list):
            msg = 'Device Specific ClI credentials must be passed as a list'
            self.discovery_specific_cred_failure(msg=msg)
        if len(cli_credentials_list) > 0:
            username_list = []
            password_list = []
            enable_password_list = []
            for cli_cred in cli_credentials_list:
                if cli_cred.get('username') and cli_cred.get('password') and cli_cred.get('enable_password'):
                    username_list.append(cli_cred.get('username'))
                    password_list.append(cli_cred.get('password'))
                    enable_password_list.append(cli_cred.get('enable_password'))
                else:
                    msg = 'username, password and enable_password must be passed toether for creating CLI credentials'
                    self.discovery_specific_cred_failure(msg=msg)
            new_object_params['userNameList'] = username_list
            new_object_params['passwordList'] = password_list
            new_object_params['enablePasswordList'] = enable_password_list
    if http_read_credential:
        if not (http_read_credential.get('password') and isinstance(http_read_credential.get('password'), str)):
            msg = 'The password for the HTTP read credential must be of string type.'
            self.discovery_specific_cred_failure(msg=msg)
        if not (http_read_credential.get('username') and isinstance(http_read_credential.get('username'), str)):
            msg = 'The username for the HTTP read credential must be of string type.'
            self.discovery_specific_cred_failure(msg=msg)
        if not (http_read_credential.get('port') and isinstance(http_read_credential.get('port'), int)):
            msg = 'The port for the HTTP read Credential must be of integer type.'
            self.discovery_specific_cred_failure(msg=msg)
        if not isinstance(http_read_credential.get('secure'), bool):
            msg = 'Secure for HTTP read Credential must be of type boolean.'
            self.discovery_specific_cred_failure(msg=msg)
        new_object_params['httpReadCredential'] = http_read_credential
    if http_write_credential:
        if not (http_write_credential.get('password') and isinstance(http_write_credential.get('password'), str)):
            msg = 'The password for the HTTP write credential must be of string type.'
            self.discovery_specific_cred_failure(msg=msg)
        if not (http_write_credential.get('username') and isinstance(http_write_credential.get('username'), str)):
            msg = 'The username for the HTTP write credential must be of string type.'
            self.discovery_specific_cred_failure(msg=msg)
        if not (http_write_credential.get('port') and isinstance(http_write_credential.get('port'), int)):
            msg = 'The port for the HTTP write Credential must be of integer type.'
            self.discovery_specific_cred_failure(msg=msg)
        if not isinstance(http_write_credential.get('secure'), bool):
            msg = 'Secure for HTTP write Credential must be of type boolean.'
            self.discovery_specific_cred_failure(msg=msg)
        new_object_params['httpWriteCredential'] = http_write_credential
    if snmp_v2_read_credential:
        if not snmp_v2_read_credential.get('desc') and isinstance(snmp_v2_read_credential.get('desc'), str):
            msg = 'Name/description for the SNMP v2 read credential must be of string type'
            self.discovery_specific_cred_failure(msg=msg)
        if not snmp_v2_read_credential.get('community') and isinstance(snmp_v2_read_credential.get('community'), str):
            msg = 'The community string must be of string type'
            self.discovery_specific_cred_failure(msg=msg)
        new_object_params['snmpROCommunityDesc'] = snmp_v2_read_credential.get('desc')
        new_object_params['snmpROCommunity'] = snmp_v2_read_credential.get('community')
        new_object_params['snmpVersion'] = 'v2'
    if snmp_v2_write_credential:
        if not snmp_v2_write_credential.get('desc') and isinstance(snmp_v2_write_credential.get('desc'), str):
            msg = 'Name/description for the SNMP v2 write credential must be of string type'
            self.discovery_specific_cred_failure(msg=msg)
        if not snmp_v2_write_credential.get('community') and isinstance(snmp_v2_write_credential.get('community'), str):
            msg = 'The community string must be of string type'
            self.discovery_specific_cred_failure(msg=msg)
        new_object_params['snmpRWCommunityDesc'] = snmp_v2_write_credential.get('desc')
        new_object_params['snmpRWCommunity'] = snmp_v2_write_credential.get('community')
        new_object_params['snmpVersion'] = 'v2'
    if snmp_v3_credential:
        if not snmp_v3_credential.get('username') and isinstance(snmp_v3_credential.get('username'), str):
            msg = 'Username of SNMP v3 protocol must be of string type'
            self.discovery_specific_cred_failure(msg=msg)
        if not snmp_v3_credential.get('snmp_mode') and isinstance(snmp_v3_credential.get('snmp_mode'), str):
            msg = 'Mode of SNMP is madantory to use SNMPv3 protocol and must be of string type'
            self.discovery_specific_cred_failure(msg=msg)
            if snmp_v3_credential.get('snmp_mode') == 'AUTHPRIV' or snmp_v3_credential.get('snmp_mode') == 'AUTHNOPRIV':
                if not snmp_v3_credential.get('auth_password') and isinstance(snmp_v3_credential.get('auth_password'), str):
                    msg = 'Authorization password must be of string type'
                    self.discovery_specific_cred_failure(msg=msg)
                if not snmp_v3_credential.get('auth_type') and isinstance(snmp_v3_credential.get('auth_type'), str):
                    msg = 'Authorization type must be of string type'
                    self.discovery_specific_cred_failure(msg=msg)
                if snmp_v3_credential.get('snmp_mode') == 'AUTHPRIV':
                    if not snmp_v3_credential.get('privacy_type') and isinstance(snmp_v3_credential.get('privacy_type'), str):
                        msg = 'Privacy type must be of string type'
                        self.discovery_specific_cred_failure(msg=msg)
                    if not snmp_v3_credential.get('privacy_password') and isinstance(snmp_v3_credential.get('privacy_password'), str):
                        msg = 'Privacy password must be of string type'
                        self.discovery_specific_cred_failure(msg=msg)
        new_object_params['snmpUserName'] = snmp_v3_credential.get('username')
        new_object_params['snmpMode'] = snmp_v3_credential.get('snmp_mode')
        new_object_params['snmpAuthPassphrase'] = snmp_v3_credential.get('auth_password')
        new_object_params['snmpAuthProtocol'] = snmp_v3_credential.get('auth_type')
        new_object_params['snmpPrivProtocol'] = snmp_v3_credential.get('privacy_type')
        new_object_params['snmpPrivPassphrase'] = snmp_v3_credential.get('privacy_password')
        new_object_params['snmpVersion'] = 'v3'
    if net_conf_port:
        new_object_params['netconfPort'] = str(net_conf_port)
    return new_object_params