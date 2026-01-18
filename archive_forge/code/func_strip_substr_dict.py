from __future__ import (absolute_import, division, print_function)
import json
import os
from ansible.module_utils.urls import open_url, ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.common.parameters import env_fallback
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import config_ipv6
def strip_substr_dict(self, odata_dict, chkstr='@odata.'):
    cp = odata_dict.copy()
    klist = cp.keys()
    for k in klist:
        if chkstr in str(k).lower():
            odata_dict.pop(k)
    return odata_dict