from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_snmpV2cWrite_credentials(self, CredentialDetails, global_credentials):
    """
        Get the current snmpV2cWrite Credential from
        Cisco DNA Center based on the provided playbook details.
        Check this API using the check_return_status.

        Parameters:
            CredentialDetails (dict) - Playbook details containing Global Device Credentials.
            global_credentials (dict) - All global device credentials details.

        Returns:
            snmpV2cWriteDetails (List) - The current snmpV2cWrite.
        """
    all_snmpV2cWrite = CredentialDetails.get('snmp_v2c_write')
    snmpV2cWrite_details = global_credentials.get('snmpV2cWrite')
    snmpV2cWriteDetails = []
    if all_snmpV2cWrite and snmpV2cWrite_details:
        for snmpV2cWriteCredential in all_snmpV2cWrite:
            snmpV2cWriteDetail = None
            snmpV2cWriteId = snmpV2cWriteCredential.get('id')
            if snmpV2cWriteId:
                snmpV2cWriteDetail = get_dict_result(snmpV2cWrite_details, 'id', snmpV2cWriteId)
                if not snmpV2cWriteDetail:
                    self.msg = 'snmpV2cWrite credential ID is invalid'
                    self.status = 'failed'
                    return self
            snmpV2cWriteOldDescription = snmpV2cWriteCredential.get('old_description')
            if snmpV2cWriteOldDescription and (not snmpV2cWriteDetail):
                snmpV2cWriteDetail = get_dict_result(snmpV2cWrite_details, 'description', snmpV2cWriteOldDescription)
                if not snmpV2cWriteDetail:
                    self.msg = 'snmpV2cWrite credential old_description is invalid '
                    self.status = 'failed'
                    return self
            snmpV2cWriteDescription = snmpV2cWriteCredential.get('description')
            if snmpV2cWriteDescription and (not snmpV2cWriteDetail):
                snmpV2cWriteDetail = get_dict_result(snmpV2cWrite_details, 'description', snmpV2cWriteDescription)
            snmpV2cWriteDetails.append(snmpV2cWriteDetail)
    return snmpV2cWriteDetails