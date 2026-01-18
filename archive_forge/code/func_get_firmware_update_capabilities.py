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
def get_firmware_update_capabilities(self):
    result = {}
    response = self.get_request(self.root_uri + self.update_uri)
    if response['ret'] is False:
        return response
    result['ret'] = True
    result['entries'] = {}
    data = response['data']
    if 'Actions' in data:
        actions = data['Actions']
        if len(actions) > 0:
            for key in actions.keys():
                action = actions.get(key)
                if 'title' in action:
                    title = action['title']
                else:
                    title = key
                result['entries'][title] = action.get('TransferProtocol@Redfish.AllowableValues', ['Key TransferProtocol@Redfish.AllowableValues not found'])
        else:
            return {'ret': 'False', 'msg': 'Actions list is empty.'}
    else:
        return {'ret': 'False', 'msg': 'Key Actions not found.'}
    return result