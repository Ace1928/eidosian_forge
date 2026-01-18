from __future__ import absolute_import, division, print_function
import datetime
import re
import time
import tarfile
from ansible.module_utils.urls import fetch_file
from ansible_collections.community.general.plugins.module_utils.redfish_utils import RedfishUtils
from ansible.module_utils.six.moves.urllib.parse import urlparse, urlunparse
def get_simple_update_status(self):
    """Issue Redfish HTTP GET to return the simple update status"""
    result = {}
    response = self.get_request(self.root_uri + self._simple_update_status_uri())
    if response['ret'] is False:
        return response
    result['ret'] = True
    data = response['data']
    result['entries'] = data
    return result