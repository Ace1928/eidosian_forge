from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
import time
import re
def preprocess_device_discovery_handle_error(self):
    """
        Method for failing discovery based on the length of list of IP Addresses passed
        for performing discovery.
        """
    self.log("IP Address list's length is longer than 1", 'ERROR')
    self.module.fail_json(msg="IP Address list's length is longer than 1", response=[])