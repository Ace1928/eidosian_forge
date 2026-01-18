from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
import time
import re
def get_creds_ids_list(self):
    """
        Retrieve the list of credentials IDs associated with class instance.

        Returns:
          The method returns the list of credentials IDs:
          - self.creds_ids_list: The list of credentials IDs associated with
                                 the class instance.
        """
    self.log('Credential Ids list passed is {0}'.format(str(self.creds_ids_list)), 'INFO')
    return self.creds_ids_list