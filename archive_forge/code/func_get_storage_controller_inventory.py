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
def get_storage_controller_inventory(self, systems_uri):
    result = {}
    controller_list = []
    controller_results = []
    properties = ['CacheSummary', 'FirmwareVersion', 'Identifiers', 'Location', 'Manufacturer', 'Model', 'Name', 'Id', 'PartNumber', 'SerialNumber', 'SpeedGbps', 'Status']
    key = 'Controllers'
    deprecated_key = 'StorageControllers'
    response = self.get_request(self.root_uri + systems_uri)
    if response['ret'] is False:
        return response
    data = response['data']
    if 'Storage' not in data:
        return {'ret': False, 'msg': 'Storage resource not found'}
    storage_uri = data['Storage']['@odata.id']
    response = self.get_request(self.root_uri + storage_uri)
    if response['ret'] is False:
        return response
    result['ret'] = True
    data = response['data']
    if data[u'Members']:
        for storage_member in data[u'Members']:
            storage_member_uri = storage_member[u'@odata.id']
            response = self.get_request(self.root_uri + storage_member_uri)
            data = response['data']
            if key in data:
                controllers_uri = data[key][u'@odata.id']
                response = self.get_request(self.root_uri + controllers_uri)
                if response['ret'] is False:
                    return response
                result['ret'] = True
                data = response['data']
                if data[u'Members']:
                    for controller_member in data[u'Members']:
                        controller_member_uri = controller_member[u'@odata.id']
                        response = self.get_request(self.root_uri + controller_member_uri)
                        if response['ret'] is False:
                            return response
                        result['ret'] = True
                        data = response['data']
                        controller_result = {}
                        for property in properties:
                            if property in data:
                                controller_result[property] = data[property]
                        controller_results.append(controller_result)
            elif deprecated_key in data:
                controller_list = data[deprecated_key]
                for controller in controller_list:
                    controller_result = {}
                    for property in properties:
                        if property in controller:
                            controller_result[property] = controller[property]
                    controller_results.append(controller_result)
            result['entries'] = controller_results
        return result
    else:
        return {'ret': False, 'msg': 'Storage resource not found'}