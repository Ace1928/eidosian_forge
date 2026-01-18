from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url, basic_auth_header
from ansible_collections.community.grafana.plugins.module_utils import base
from ansible.module_utils.six.moves.urllib.parse import quote
def create_org(self, name):
    url = '/api/orgs'
    org = dict(name=name)
    self._send_request(url, data=org, headers=self.headers, method='POST')
    return self.get_actual_org(name)