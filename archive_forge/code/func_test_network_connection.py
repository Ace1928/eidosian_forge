from __future__ import (absolute_import, division, print_function)
import json
import os
import time
from ansible.module_utils.urls import open_url, ConnectionError, SSLValidationError
from ansible.module_utils.common.parameters import env_fallback
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import config_ipv6
def test_network_connection(self, share_address, share_path, share_type, share_user=None, share_password=None, share_domain=None):
    job_type = {'Id': 56, 'Name': 'ValidateNWFileShare_Task'}
    params = [{'Key': 'checkPathOnly', 'Value': 'false'}, {'Key': 'shareType', 'Value': share_type}, {'Key': 'ShareNetworkFilePath', 'Value': share_path}, {'Key': 'shareAddress', 'Value': share_address}, {'Key': 'testShareWriteAccess', 'Value': 'true'}]
    if share_user is not None:
        params.append({'Key': 'UserName', 'Value': share_user})
    if share_password is not None:
        params.append({'Key': 'Password', 'Value': share_password})
    if share_domain is not None:
        params.append({'Key': 'domainName', 'Value': share_domain})
    job_response = self.job_submission('Validate Share', 'Validate Share', [], params, job_type)
    return job_response