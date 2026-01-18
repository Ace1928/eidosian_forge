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
def set_hostinterface_attributes(self, hostinterface_config, hostinterface_id=None):
    if hostinterface_config is None:
        return {'ret': False, 'msg': 'Must provide hostinterface_config for SetHostInterface command'}
    response = self.get_request(self.root_uri + self.manager_uri)
    if response['ret'] is False:
        return response
    data = response['data']
    hostinterfaces_uri = data.get('HostInterfaces', {}).get('@odata.id')
    if hostinterfaces_uri is None:
        return {'ret': False, 'msg': 'HostInterface resource not found'}
    response = self.get_request(self.root_uri + hostinterfaces_uri)
    if response['ret'] is False:
        return response
    data = response['data']
    uris = [a.get('@odata.id') for a in data.get('Members', []) if a.get('@odata.id')]
    if hostinterface_id:
        matching_hostinterface_uris = [uri for uri in uris if hostinterface_id in uri.split('/')[-1]]
    if hostinterface_id and matching_hostinterface_uris:
        hostinterface_uri = list.pop(matching_hostinterface_uris)
    elif hostinterface_id and (not matching_hostinterface_uris):
        return {'ret': False, 'msg': 'HostInterface ID %s not present.' % hostinterface_id}
    elif len(uris) == 1:
        hostinterface_uri = list.pop(uris)
    else:
        return {'ret': False, 'msg': 'HostInterface ID not defined and multiple interfaces detected.'}
    resp = self.patch_request(self.root_uri + hostinterface_uri, hostinterface_config, check_pyld=True)
    if resp['ret'] and resp['changed']:
        resp['msg'] = 'Modified host interface'
    return resp