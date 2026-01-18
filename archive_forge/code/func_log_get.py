from __future__ import (absolute_import, division, print_function)
import os
import time
import traceback
from ansible.module_utils._text import to_text
import json
from ansible_collections.fortinet.fortios.plugins.module_utils.common.type_utils import underscore_to_hyphen
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.secret_field import is_secret_field
def log_get(self, url, parameters=None):
    slash_index = url.find('/')
    full_url = self.log_url(url[:slash_index], url[slash_index + 1:])
    http_status, result_data = self._conn.send_request(url=full_url, params=parameters, method='GET')
    return self.formatresponse(result_data, http_status)