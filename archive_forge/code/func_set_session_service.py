from __future__ import absolute_import, division, print_function
import json
import os
import random
import string
import gzip
from io import BytesIO
from ansible.module_utils.urls import open_url
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.six import text_type
from ansible.module_utils.six.moves import http_client
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.ansible_release import __version__ as ansible_version
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def set_session_service(self, sessions_config):
    if sessions_config is None:
        return {'ret': False, 'msg': 'Must provide sessions_config for SetSessionService command'}
    resp = self.patch_request(self.root_uri + self.session_service_uri, sessions_config, check_pyld=True)
    if resp['ret'] and resp['changed']:
        resp['msg'] = 'Modified session service'
    return resp