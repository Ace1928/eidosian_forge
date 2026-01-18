from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_have_device_credentials(self, CredentialDetails):
    """
        Get the current Global Device Credentials from
        Cisco DNA Center based on the provided playbook details.
        Check this API using the check_return_status.

        Parameters:
            CredentialDetails (dict) - Playbook details containing Global Device Credentials.

        Returns:
            self - The current object with updated information.
        """
    global_credentials = self.get_global_credentials_params()
    cliDetails = self.get_cli_credentials(CredentialDetails, global_credentials)
    snmpV2cReadDetails = self.get_snmpV2cRead_credentials(CredentialDetails, global_credentials)
    snmpV2cWriteDetails = self.get_snmpV2cWrite_credentials(CredentialDetails, global_credentials)
    httpsReadDetails = self.get_httpsRead_credentials(CredentialDetails, global_credentials)
    httpsWriteDetails = self.get_httpsWrite_credentials(CredentialDetails, global_credentials)
    snmpV3Details = self.get_snmpV3_credentials(CredentialDetails, global_credentials)
    self.have.update({'globalCredential': {}})
    if cliDetails:
        cliCredential = self.get_cli_params(cliDetails)
        self.have.get('globalCredential').update({'cliCredential': cliCredential})
    if snmpV2cReadDetails:
        snmpV2cRead = self.get_snmpV2cRead_params(snmpV2cReadDetails)
        self.have.get('globalCredential').update({'snmpV2cRead': snmpV2cRead})
    if snmpV2cWriteDetails:
        snmpV2cWrite = self.get_snmpV2cWrite_params(snmpV2cWriteDetails)
        self.have.get('globalCredential').update({'snmpV2cWrite': snmpV2cWrite})
    if httpsReadDetails:
        httpsRead = self.get_httpsRead_params(httpsReadDetails)
        self.have.get('globalCredential').update({'httpsRead': httpsRead})
    if httpsWriteDetails:
        httpsWrite = self.get_httpsWrite_params(httpsWriteDetails)
        self.have.get('globalCredential').update({'httpsWrite': httpsWrite})
    if snmpV3Details:
        snmpV3 = self.get_snmpV3_params(snmpV3Details)
        self.have.get('globalCredential').update({'snmpV3': snmpV3})
    self.log('Global device credential details: {0}'.format(self.have.get('globalCredential')), 'DEBUG')
    self.msg = 'Collected the Global Device Credential Details from the Cisco DNA Center'
    self.status = 'success'
    return self