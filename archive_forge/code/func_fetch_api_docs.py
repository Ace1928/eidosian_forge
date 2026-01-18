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
def fetch_api_docs(self):
    try:
        tmp_dir = os.path.split(DEFAULT_LOCAL_TMP)[0]
        tmp_file = os.path.join(tmp_dir, 'netbox_api_dump.json')
        with open(tmp_file) as file:
            cache = json.load(file)
        cached_api_version = '.'.join(cache['info']['version'].split('.')[:2])
    except Exception:
        cached_api_version = None
        cache = None
    status = self._fetch_information(self.api_endpoint + '/api/status')
    netbox_api_version = '.'.join(status['netbox-version'].split('.')[:2])
    if version.parse(netbox_api_version) >= version.parse('3.5.0'):
        endpoint_url = self.api_endpoint + '/api/schema/?format=json'
    else:
        endpoint_url = self.api_endpoint + '/api/docs/?format=openapi'
    if cache and cached_api_version == netbox_api_version:
        openapi = cache
    else:
        openapi = self._fetch_information(endpoint_url)
        try:
            with open(tmp_file, 'w') as file:
                json.dump(openapi, file)
        except Exception:
            pass
    self.api_version = version.parse(netbox_api_version)
    if self.api_version >= version.parse('3.5.0'):
        self.allowed_device_query_parameters = [p['name'] for p in openapi['paths']['/api/dcim/devices/']['get']['parameters']]
        self.allowed_vm_query_parameters = [p['name'] for p in openapi['paths']['/api/virtualization/virtual-machines/']['get']['parameters']]
    else:
        self.allowed_device_query_parameters = [p['name'] for p in openapi['paths']['/dcim/devices/']['get']['parameters']]
        self.allowed_vm_query_parameters = [p['name'] for p in openapi['paths']['/virtualization/virtual-machines/']['get']['parameters']]