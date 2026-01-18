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
def get_license_url(self):
    """
        Retrieves the license URL for the current user.

        :return: The license URL as a string.
        """
    v1_resp = get_dynamic_uri(self.idrac, REDFISH)
    license_service_url = v1_resp.get('LicenseService', {}).get(ODATA, {})
    license_service_resp = get_dynamic_uri(self.idrac, license_service_url)
    license_url = license_service_resp.get('Licenses', {}).get(ODATA, {})
    return license_url