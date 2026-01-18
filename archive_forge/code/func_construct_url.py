from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_text
from ansible.module_utils.six.moves import http_client as httplib
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import quote
import logging
def construct_url(api_base_url, end_point):
    return '{baseurl}/{endpoint}'.format(baseurl=api_base_url.rstrip('/'), endpoint=end_point.lstrip('/'))