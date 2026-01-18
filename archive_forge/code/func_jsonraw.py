from __future__ import (absolute_import, division, print_function)
import os
import time
import traceback
from ansible.module_utils._text import to_text
import json
from ansible_collections.fortinet.fortios.plugins.module_utils.common.type_utils import underscore_to_hyphen
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.secret_field import is_secret_field
def jsonraw(self, method, path, data, specific_params, vdom=None, parameters=None):
    url = path
    bvdom = False
    if vdom:
        if vdom == 'global':
            url += '?global=1'
        else:
            url += '?vdom=' + vdom
        bvdom = True
    if specific_params:
        if bvdom:
            url += '&'
        else:
            url += '?'
        url += specific_params
    if method == 'GET':
        http_status, result_data = self._conn.send_request(url=url, method='GET', params=parameters)
    else:
        http_status, result_data = self._conn.send_request(url=url, method=method, data=json.dumps(data), params=parameters)
    return self.formatresponse(result_data, http_status, vdom=vdom)