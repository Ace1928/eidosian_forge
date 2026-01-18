from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_snmpV2cRead_params(self, snmpV2cReadDetails):
    """
        Format the snmpV2cRead parameters for the snmpV2cRead
        credential configuration in Cisco DNA Center.

        Parameters:
            snmpV2cReadDetails (list of dict) - Cisco DNA Center
            Details containing snmpV2cRead Credentials.

        Returns:
            snmpV2cRead (list of dict) - Processed snmpV2cRead credential
            data in the format suitable for the Cisco DNA Center config.
        """
    snmpV2cRead = []
    for item in snmpV2cReadDetails:
        if item is None:
            snmpV2cRead.append(None)
        else:
            value = {'description': item.get('description'), 'id': item.get('id')}
            snmpV2cRead.append(value)
    return snmpV2cRead