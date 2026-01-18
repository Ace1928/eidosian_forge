from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url, basic_auth_header
from ansible_collections.community.grafana.plugins.module_utils import base
from ansible.module_utils.six.moves.urllib.parse import quote
def get_user_from_login(self, login):
    url = '/api/users/lookup?loginOrEmail={login}'.format(login=quote(login))
    return self._send_request(url, headers=self.headers, method='GET')