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
def get_cpu_inventory(self, systems_uri):
    result = {}
    cpu_list = []
    cpu_results = []
    key = 'Processors'
    properties = ['Id', 'Name', 'Manufacturer', 'Model', 'MaxSpeedMHz', 'ProcessorArchitecture', 'TotalCores', 'TotalThreads', 'Status']
    response = self.get_request(self.root_uri + systems_uri)
    if response['ret'] is False:
        return response
    result['ret'] = True
    data = response['data']
    if key not in data:
        return {'ret': False, 'msg': 'Key %s not found' % key}
    processors_uri = data[key]['@odata.id']
    response = self.get_request(self.root_uri + processors_uri)
    if response['ret'] is False:
        return response
    result['ret'] = True
    data = response['data']
    for cpu in data[u'Members']:
        cpu_list.append(cpu[u'@odata.id'])
    for c in cpu_list:
        cpu = {}
        uri = self.root_uri + c
        response = self.get_request(uri)
        if response['ret'] is False:
            return response
        data = response['data']
        for property in properties:
            if property in data:
                cpu[property] = data[property]
        cpu_results.append(cpu)
    result['entries'] = cpu_results
    return result