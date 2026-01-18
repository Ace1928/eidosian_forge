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
def fetch_role_related(self, related, role_id):
    """
        Fetch the list of related items for the given role.
        The url comes from the 'related' field of the role.
        """
    results = []
    try:
        url = _urljoin(self.api_server, self.available_api_versions['v1'], 'roles', role_id, related, '?page_size=50')
        data = self._call_galaxy(url)
        results = data['results']
        done = data.get('next_link', None) is None
        url_info = urlparse(self.api_server)
        base_url = '%s://%s/' % (url_info.scheme, url_info.netloc)
        while not done:
            url = _urljoin(base_url, data['next_link'])
            data = self._call_galaxy(url)
            results += data['results']
            done = data.get('next_link', None) is None
    except Exception as e:
        display.warning('Unable to retrieve role (id=%s) data (%s), but this is not fatal so we continue: %s' % (role_id, related, to_text(e)))
    return results