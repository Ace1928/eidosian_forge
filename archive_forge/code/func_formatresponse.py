from __future__ import (absolute_import, division, print_function)
import os
import time
import traceback
from ansible.module_utils._text import to_text
import json
from ansible_collections.fortinet.fortios.plugins.module_utils.common.type_utils import underscore_to_hyphen
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.secret_field import is_secret_field
def formatresponse(self, res, http_status=500, vdom=None):
    if vdom == 'global':
        resp = self.__to_local(to_text(res), http_status, True)[0]
        resp['vdom'] = 'global'
    else:
        resp = self.__to_local(to_text(res), http_status, False)
    return resp