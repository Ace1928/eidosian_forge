from __future__ import absolute_import, division, print_function
from copy import deepcopy
import re
import os
import ast
import datetime
import shutil
import tempfile
from ansible.module_utils.basic import json
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.six import PY3
from ansible.module_utils.six.moves import filterfalse
from ansible.module_utils.six.moves.urllib.parse import urlencode, urljoin
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_native, to_text
from ansible.module_utils.connection import Connection
from ansible_collections.cisco.mso.plugins.module_utils.constants import NDO_API_VERSION_PATH_FORMAT
def delete_keys_from_dict(self, dict_to_sanitize, keys):
    copy = deepcopy(dict_to_sanitize)
    for k, v in copy.items():
        if k in keys:
            del dict_to_sanitize[k]
        elif isinstance(v, dict):
            dict_to_sanitize[k] = self.delete_keys_from_dict(v, keys)
        elif isinstance(v, list):
            for index, item in enumerate(v):
                if isinstance(item, dict):
                    dict_to_sanitize[k][index] = self.delete_keys_from_dict(item, keys)
    return dict_to_sanitize