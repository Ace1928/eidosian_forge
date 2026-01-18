from __future__ import absolute_import, division, print_function
import json
import os
import uuid
from ansible.module_utils.urls import open_url
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.six.moves.urllib.parse import urlparse
def manage_power_mode(self, command):
    key = 'PowerState'
    resource_uri = self.get_uri_with_slot_number_query_param(self.root_uri)
    payloads = {'PowerModeNormal': 2, 'PowerModeLow': 4}
    response = self.get_request(resource_uri)
    if 'etag' not in response['headers']:
        return {'ret': False, 'msg': 'Etag not found in response.'}
    etag = response['headers']['etag']
    if response['ret'] is False:
        return response
    data = response['data']
    if key not in data:
        return {'ret': False, 'msg': 'Key %s not found' % key}
    if 'ID' not in data[key]:
        return {'ret': False, 'msg': 'PowerState for resource has no ID.'}
    if command in payloads.keys():
        current_power_state = data[key]['ID']
        if current_power_state == payloads[command]:
            return {'ret': True, 'changed': False}
        if self.module.check_mode:
            return {'ret': True, 'changed': True, 'msg': 'Update not performed in check mode.'}
        payload = {'PowerState': {'ID': payloads[command]}}
        response = self.put_request(resource_uri, payload, etag)
        if response['ret'] is False:
            return response
    else:
        return {'ret': False, 'msg': 'Invalid command: ' + command}
    return {'ret': True}