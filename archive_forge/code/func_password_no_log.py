from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import apply_diff_key, job_tracking
def password_no_log(attributes):
    if isinstance(attributes, dict):
        netdict = attributes.get('NetworkBootIsoModel')
        if isinstance(netdict, dict):
            sharedet = netdict.get('ShareDetail')
            if isinstance(sharedet, dict) and 'Password' in sharedet:
                sharedet['Password'] = 'VALUE_SPECIFIED_IN_NO_LOG_PARAMETER'