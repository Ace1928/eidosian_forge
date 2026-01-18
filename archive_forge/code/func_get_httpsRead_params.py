from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_httpsRead_params(self, httpsReadDetails):
    """
        Format the httpsRead parameters for the httpsRead
        credential configuration in Cisco DNA Center.

        Parameters:
            httpsReadDetails (list of dict) - Cisco DNA Center
            Details containing httpsRead Credentials.

        Returns:
            httpsRead (list of dict) - Processed httpsRead credential
            data in the format suitable for the Cisco DNA Center config.
        """
    httpsRead = []
    for item in httpsReadDetails:
        if item is None:
            httpsRead.append(None)
        else:
            value = {'description': item.get('description'), 'username': item.get('username'), 'port': item.get('port'), 'id': item.get('id')}
            httpsRead.append(value)
    return httpsRead