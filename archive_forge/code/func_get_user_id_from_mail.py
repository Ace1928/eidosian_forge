from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url, basic_auth_header
from ansible.module_utils._text import to_text
from ansible_collections.community.grafana.plugins.module_utils import base
from ansible.module_utils.six.moves.urllib.parse import quote
def get_user_id_from_mail(self, email):
    url = '/api/users/lookup?loginOrEmail={email}'.format(email=quote(email))
    user = self._send_request(url, headers=self.headers, method='GET')
    if user is None:
        self._module.fail_json(failed=True, msg="User '%s' does not exists" % email)
    return user.get('id')