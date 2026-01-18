from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url, basic_auth_header
from ansible_collections.community.grafana.plugins.module_utils import base
from ansible.module_utils.six.moves.urllib.parse import quote
from ansible.module_utils._text import to_text
def organization_by_name(self, org_name):
    url = '/api/user/orgs'
    organizations = self._send_request(url, headers=self.headers, method='GET')
    orga = next((org for org in organizations if org['name'] == org_name))
    if orga:
        return orga['orgId']
    self._module.fail_json(failed=True, msg="Current user isn't member of organization: %s" % org_name)