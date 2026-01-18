from __future__ import (absolute_import, division, print_function)
import os
import time
import traceback
from ansible.module_utils._text import to_text
import json
from ansible_collections.fortinet.fortios.plugins.module_utils.common.type_utils import underscore_to_hyphen
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.secret_field import is_secret_field
def cmdb_url(self, path, name, vdom=None, mkey=None):
    url = '/api/v2/cmdb/' + path + '/' + name
    if mkey:
        url = url + '/' + urlencoding.quote(str(mkey), safe='')
    if vdom:
        if vdom == 'global':
            url += '?global=1'
        else:
            url += '?vdom=' + vdom
    return url