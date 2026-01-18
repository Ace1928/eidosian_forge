from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url, basic_auth_header
from ansible.module_utils._text import to_text
from ansible_collections.community.grafana.plugins.module_utils import base
from ansible.module_utils.six.moves.urllib.parse import quote
def delete_team(self, team_id):
    url = '/api/teams/{team_id}'.format(team_id=team_id)
    response = self._send_request(url, headers=self.headers, method='DELETE')
    return response