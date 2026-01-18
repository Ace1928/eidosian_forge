from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_global_credentials_params(self):
    """
        Get the current Global Device Credentials from Cisco DNA Center.

        Parameters:
            self - The current object details.

        Returns:
            global_credentials (dict) - All global device credentials details.
        """
    try:
        global_credentials = self.dnac._exec(family='discovery', function='get_all_global_credentials_v2')
        global_credentials = global_credentials.get('response')
        self.log('All global device credentials details: {0}'.format(global_credentials), 'DEBUG')
    except Exception as exec:
        self.log('Exception occurred while getting global device credentials: {0}'.format(exec), 'CRITICAL')
        return None
    return global_credentials