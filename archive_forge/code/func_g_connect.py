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
def g_connect(versions):
    """
    Wrapper to lazily initialize connection info to Galaxy and verify the API versions required are available on the
    endpoint.

    :param versions: A list of API versions that the function supports.
    """

    def decorator(method):

        def wrapped(self, *args, **kwargs):
            if not self._available_api_versions:
                display.vvvv('Initial connection to galaxy_server: %s' % self.api_server)
                n_url = self.api_server
                error_context_msg = 'Error when finding available api versions from %s (%s)' % (self.name, n_url)
                if self.api_server == 'https://galaxy.ansible.com' or self.api_server == 'https://galaxy.ansible.com/':
                    n_url = 'https://galaxy.ansible.com/api/'
                try:
                    data = self._call_galaxy(n_url, method='GET', error_context_msg=error_context_msg, cache=True)
                except (AnsibleError, GalaxyError, ValueError, KeyError) as err:
                    if n_url.endswith('/api') or n_url.endswith('/api/'):
                        raise
                    n_url = _urljoin(n_url, '/api/')
                    try:
                        data = self._call_galaxy(n_url, method='GET', error_context_msg=error_context_msg, cache=True)
                    except GalaxyError as new_err:
                        if new_err.http_code == 404:
                            raise err
                        raise
                if 'available_versions' not in data:
                    raise AnsibleError("Tried to find galaxy API root at %s but no 'available_versions' are available on %s" % (n_url, self.api_server))
                self.api_server = n_url
                available_versions = data.get('available_versions', {u'v1': u'v1/'})
                if list(available_versions.keys()) == [u'v1']:
                    available_versions[u'v2'] = u'v2/'
                self._available_api_versions = available_versions
                display.vvvv("Found API version '%s' with Galaxy server %s (%s)" % (', '.join(available_versions.keys()), self.name, self.api_server))
            available_versions = set(self._available_api_versions.keys())
            common_versions = set(versions).intersection(available_versions)
            if not common_versions:
                raise AnsibleError("Galaxy action %s requires API versions '%s' but only '%s' are available on %s %s" % (method.__name__, ', '.join(versions), ', '.join(available_versions), self.name, self.api_server))
            return method(self, *args, **kwargs)
        return wrapped
    return decorator