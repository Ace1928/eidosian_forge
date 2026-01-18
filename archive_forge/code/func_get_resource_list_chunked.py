from __future__ import absolute_import, division, print_function
import json
import uuid
import math
import os
import datetime
from copy import deepcopy
from functools import partial
from sys import version as python_version
from threading import Thread
from typing import Iterable
from itertools import chain
from collections import defaultdict
from ipaddress import ip_interface
from ansible.constants import DEFAULT_LOCAL_TMP
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable, Cacheable
from ansible.module_utils.ansible_release import __version__ as ansible_version
from ansible.errors import AnsibleError
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib import error as urllib_error
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.six import raise_from
def get_resource_list_chunked(self, api_url, query_key, query_values):
    if not isinstance(query_values, list):
        query_values = list(query_values)

    def query_string(value, separator='&'):
        return separator + query_key + '=' + str(value)
    largest_value = str(max(query_values, default=0))
    length_per_value = len(query_string(largest_value))
    chunk_size = math.floor((self.max_uri_length - len(api_url)) / length_per_value)
    if chunk_size < 1:
        chunk_size = 1
    if self.api_version in specifiers.SpecifierSet('~=2.6.0'):
        chunk_size = 1
    resources = []
    for i in range(0, len(query_values), chunk_size):
        chunk = query_values[i:i + chunk_size]
        url = api_url
        for value in chunk:
            url += query_string(value, '&' if '?' in url else '?')
        resources.extend(self.get_resource_list(url))
    return resources