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
def lookup_labels(self, labels, label_type, ignore_not_found_error=False):
    """Look up labels and return their ids (create if necessary)"""
    if labels is None:
        return None
    ids = []
    for label in labels:
        label_obj = self.get_obj('labels', displayName=label)
        if not label_obj:
            label_obj = self.create_label(label, label_type)
        if 'id' not in label_obj and (not ignore_not_found_error):
            self.fail_json(msg="Label lookup failed for label '{0}': {1}".format(label, label_obj))
        elif 'id' not in label_obj and ignore_not_found_error:
            self.module.warn("Label lookup failed for label '{0}': {1}".format(label, label_obj))
            return ids
        ids.append(label_obj.get('id'))
    return ids