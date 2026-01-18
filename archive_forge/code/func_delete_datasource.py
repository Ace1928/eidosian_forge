from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import quote
from ansible.module_utils.urls import fetch_url, basic_auth_header
from ansible_collections.community.grafana.plugins.module_utils import base
def delete_datasource(self, name):
    url = '/api/datasources/name/%s' % quote(name, safe='')
    self._send_request(url, headers=self.headers, method='DELETE')