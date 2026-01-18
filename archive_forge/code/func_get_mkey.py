from __future__ import (absolute_import, division, print_function)
import os
import time
import traceback
from ansible.module_utils._text import to_text
import json
from ansible_collections.fortinet.fortios.plugins.module_utils.common.type_utils import underscore_to_hyphen
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.secret_field import is_secret_field
def get_mkey(self, path, name, data, vdom=None):
    keyname = self.get_mkeyname(path, name, vdom)
    if not keyname:
        return None
    else:
        try:
            mkey = data[keyname]
        except KeyError:
            return None
    return mkey