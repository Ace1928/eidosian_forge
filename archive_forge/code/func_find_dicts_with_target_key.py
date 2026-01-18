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
def find_dicts_with_target_key(self, target_dict, target, replace, result=None):
    if result is None:
        result = []
    for key, value in target_dict.items():
        if key == target:
            result.append(target_dict)
        if isinstance(value, dict):
            self.find_dicts_with_target_key(value, target, replace, result)
        if isinstance(value, list):
            for entry in value:
                if isinstance(entry, dict):
                    self.find_dicts_with_target_key(entry, target, replace, result)
    return result