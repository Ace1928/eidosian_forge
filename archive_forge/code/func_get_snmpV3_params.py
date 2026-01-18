from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_snmpV3_params(self, snmpV3Details):
    """
        Format the snmpV3 parameters for the snmpV3 credential configuration in Cisco DNA Center.

        Parameters:
            snmpV3Details (list of dict) - Cisco DNA Center details containing snmpV3 Credentials.

        Returns:
            snmpV3 (list of dict) - Processed snmpV3 credential
            data in the format suitable for the Cisco DNA Center config.
        """
    snmpV3 = []
    for item in snmpV3Details:
        if item is None:
            snmpV3.append(None)
        else:
            value = {'username': item.get('username'), 'description': item.get('description'), 'snmpMode': item.get('snmpMode'), 'id': item.get('id')}
            if value.get('snmpMode') == 'AUTHNOPRIV':
                value['authType'] = item.get('authType')
            elif value.get('snmpMode') == 'AUTHPRIV':
                value.update({'authType': item.get('authType'), 'privacyType': item.get('privacyType')})
            snmpV3.append(value)
    return snmpV3