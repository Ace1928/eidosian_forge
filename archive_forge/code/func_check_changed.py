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
def check_changed(self):
    """Check if changed by comparing new values from existing"""
    existing = self.existing
    if 'password' in existing:
        existing['password'] = self.sent.get('password')
    existing = self.remove_keys_from_dict_when_value_empty(existing)
    self.stdout = json.dumps(existing)
    return not issubset(self.sent, existing)