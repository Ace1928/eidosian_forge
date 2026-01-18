from __future__ import (absolute_import, division, print_function)
import json
import os
import base64
from urllib.error import HTTPError, URLError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI, idrac_auth_params
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.compat.version import LooseVersion
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import (
def get_share_details(self):
    """
        Retrieves the share details from the given module.

        Args:
            module (object): The module object containing the share parameters.

        Returns:
            dict: A dictionary containing the share details with the following keys:
                - IPAddress (str): The IP address of the share.
                - ShareName (str): The name of the share.
                - UserName (str): The username for accessing the share.
                - Password (str): The password for accessing the share.
        """
    share_details = {}
    share_details['IPAddress'] = self.module.params.get('share_parameters').get('ip_address')
    share_details['ShareName'] = self.module.params.get('share_parameters').get('share_name')
    share_details['UserName'] = self.module.params.get('share_parameters').get('username')
    share_details['Password'] = self.module.params.get('share_parameters').get('password')
    return share_details