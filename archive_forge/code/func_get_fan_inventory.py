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
def get_fan_inventory(self):
    result = {}
    fan_results = []
    key = 'Thermal'
    properties = ['Name', 'FanName', 'Reading', 'ReadingUnits', 'Status']
    for chassis_uri in self.chassis_uris:
        response = self.get_request(self.root_uri + chassis_uri)
        if response['ret'] is False:
            return response
        result['ret'] = True
        data = response['data']
        if key in data:
            thermal_uri = data[key]['@odata.id']
            response = self.get_request(self.root_uri + thermal_uri)
            if response['ret'] is False:
                return response
            result['ret'] = True
            data = response['data']
            if u'Fans' in data:
                for device in data[u'Fans']:
                    fan = {}
                    for property in properties:
                        if property in device:
                            fan[property] = device[property]
                    fan_results.append(fan)
            else:
                return {'ret': False, 'msg': 'No Fans present'}
    result['entries'] = fan_results
    return result