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
def delete_volumes(self, storage_subsystem_id, volume_ids):
    response = self.get_request(self.root_uri + self.systems_uri)
    if response['ret'] is False:
        return response
    data = response['data']
    storage_uri = data.get('Storage', {}).get('@odata.id')
    if storage_uri is None:
        return {'ret': False, 'msg': 'Storage resource not found'}
    response = self.get_request(self.root_uri + storage_uri)
    if response['ret'] is False:
        return response
    data = response['data']
    self.storage_subsystems_uris = [i['@odata.id'] for i in response['data'].get('Members', [])]
    if not self.storage_subsystems_uris:
        return {'ret': False, 'msg': "StorageCollection's Members array is either empty or missing"}
    self.storage_subsystem_uri = ''
    for storage_subsystem_uri in self.storage_subsystems_uris:
        if storage_subsystem_uri.split('/')[-2] == storage_subsystem_id:
            self.storage_subsystem_uri = storage_subsystem_uri
    if not self.storage_subsystem_uri:
        return {'ret': False, 'msg': 'Provided Storage Subsystem ID %s does not exist on the server' % storage_subsystem_id}
    response = self.get_request(self.root_uri + self.storage_subsystem_uri)
    if response['ret'] is False:
        return response
    data = response['data']
    response = self.get_request(self.root_uri + data['Volumes']['@odata.id'])
    if response['ret'] is False:
        return response
    data = response['data']
    self.volume_uris = [i['@odata.id'] for i in response['data'].get('Members', [])]
    if not self.volume_uris:
        return {'ret': True, 'changed': False, 'msg': "VolumeCollection's Members array is either empty or missing"}
    for volume in self.volume_uris:
        if volume.split('/')[-1] in volume_ids:
            response = self.delete_request(self.root_uri + volume)
            if response['ret'] is False:
                return response
    return {'ret': True, 'changed': True, 'msg': 'The following volumes were deleted: %s' % str(volume_ids)}