from __future__ import (absolute_import, division, print_function)
import json
import re
import sys
import datetime
import time
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six.moves.urllib.parse import urlencode
def fetch_state(self, resource):
    self.module.debug('fetch_state of resource: %s' % resource['id'])
    response = self.get(path=self.api_path + '/%s' % resource['id'])
    if response.status_code == 404:
        return 'absent'
    if not response.ok:
        msg = 'Error during state fetching: (%s) %s' % (response.status_code, response.json)
        self.module.fail_json(msg=msg)
    try:
        self.module.debug('Resource %s in state: %s' % (resource['id'], response.json['status']))
        return response.json['status']
    except KeyError:
        self.module.fail_json(msg='Could not fetch state in %s' % response.json)