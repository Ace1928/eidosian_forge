from __future__ import absolute_import, division, print_function
import datetime
import re
import time
import tarfile
from ansible.module_utils.urls import fetch_file
from ansible_collections.community.general.plugins.module_utils.redfish_utils import RedfishUtils
from ansible.module_utils.six.moves.urllib.parse import urlparse, urlunparse
def firmware_activate(self, update_opts):
    """Perform FWActivate using Redfish HTTP API."""
    creds = update_opts.get('update_creds')
    payload = {}
    if creds:
        if creds.get('username'):
            payload['Username'] = creds.get('username')
        if creds.get('password'):
            payload['Password'] = creds.get('password')
    response = self.get_request(self.root_uri + self._update_uri())
    if response['ret'] is False:
        return response
    data = response['data']
    if 'Actions' not in data:
        return {'ret': False, 'msg': 'Service does not support FWActivate'}
    response = self.post_request(self.root_uri + self._firmware_activate_uri(), payload)
    if response['ret'] is False:
        return response
    return {'ret': True, 'changed': True, 'msg': 'FWActivate requested'}