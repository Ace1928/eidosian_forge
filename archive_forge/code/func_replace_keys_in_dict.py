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
def replace_keys_in_dict(self, target, replace, target_dict=None):
    if target_dict is None:
        target_dict = self.existing
    key_list = self.find_dicts_with_target_key(target_dict, target, replace)
    for item in key_list:
        item[replace] = item.get(target)
        del item[target]