from __future__ import absolute_import, division, print_function
import json
import os
import re
import shutil
import sys
import tempfile
from ansible.module_utils.basic import AnsibleModule, sanitize_keys
from ansible.module_utils.six import PY2, PY3, binary_type, iteritems, string_types
from ansible.module_utils.six.moves.urllib.parse import urlencode, urlsplit
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.compat.datetime import utcnow, utcfromtimestamp
from ansible.module_utils.six.moves.collections_abc import Mapping, Sequence
from ansible.module_utils.urls import fetch_url, get_response_filename, parse_content_type, prepare_multipart, url_argument_spec
def form_urlencoded(body):
    """ Convert data into a form-urlencoded string """
    if isinstance(body, string_types):
        return body
    if isinstance(body, (Mapping, Sequence)):
        result = []
        for key, values in kv_list(body):
            if isinstance(values, string_types) or not isinstance(values, (Mapping, Sequence)):
                values = [values]
            for value in values:
                if value is not None:
                    result.append((to_text(key), to_text(value)))
        return urlencode(result, doseq=True)
    return body