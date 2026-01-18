from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url, basic_auth_header
from ansible.module_utils._text import to_text
from ansible_collections.community.grafana.plugins.module_utils import base
from ansible.module_utils.six.moves.urllib.parse import quote
def add_team_member(self, team_id, email):
    url = '/api/teams/{team_id}/members'.format(team_id=team_id)
    data = {'userId': self.get_user_id_from_mail(email)}
    self._send_request(url, data=data, headers=self.headers, method='POST')