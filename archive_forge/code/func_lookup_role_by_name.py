from __future__ import (absolute_import, division, print_function)
import collections
import datetime
import functools
import hashlib
import json
import os
import stat
import tarfile
import time
import threading
from http import HTTPStatus
from http.client import BadStatusLine, IncompleteRead
from urllib.error import HTTPError, URLError
from urllib.parse import quote as urlquote, urlencode, urlparse, parse_qs, urljoin
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.galaxy.user_agent import user_agent
from ansible.module_utils.api import retry_with_delays_and_condition
from ansible.module_utils.api import generate_jittered_backoff
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.urls import open_url, prepare_multipart
from ansible.utils.display import Display
from ansible.utils.hashing import secure_hash_s
from ansible.utils.path import makedirs_safe
@g_connect(['v1'])
def lookup_role_by_name(self, role_name, notify=True):
    """
        Find a role by name.
        """
    role_name = to_text(urlquote(to_bytes(role_name)))
    try:
        parts = role_name.split('.')
        user_name = '.'.join(parts[0:-1])
        role_name = parts[-1]
        if notify:
            display.display("- downloading role '%s', owned by %s" % (role_name, user_name))
    except Exception:
        raise AnsibleError('Invalid role name (%s). Specify role as format: username.rolename' % role_name)
    url = _urljoin(self.api_server, self.available_api_versions['v1'], 'roles', '?owner__username=%s&name=%s' % (user_name, role_name))
    data = self._call_galaxy(url)
    if len(data['results']) != 0:
        return data['results'][0]
    return None