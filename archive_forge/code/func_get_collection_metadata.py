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
@g_connect(['v2', 'v3'])
def get_collection_metadata(self, namespace, name):
    """
        Gets the collection information from the Galaxy server about a specific Collection.

        :param namespace: The collection namespace.
        :param name: The collection name.
        return: CollectionMetadata about the collection.
        """
    if 'v3' in self.available_api_versions:
        api_path = self.available_api_versions['v3']
        field_map = [('created_str', 'created_at'), ('modified_str', 'updated_at')]
    else:
        api_path = self.available_api_versions['v2']
        field_map = [('created_str', 'created'), ('modified_str', 'modified')]
    info_url = _urljoin(self.api_server, api_path, 'collections', namespace, name, '/')
    error_context_msg = 'Error when getting the collection info for %s.%s from %s (%s)' % (namespace, name, self.name, self.api_server)
    data = self._call_galaxy(info_url, error_context_msg=error_context_msg)
    metadata = {}
    for name, api_field in field_map:
        metadata[name] = data.get(api_field, None)
    return CollectionMetadata(namespace, name, **metadata)