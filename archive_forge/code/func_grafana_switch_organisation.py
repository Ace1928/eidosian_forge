from __future__ import absolute_import, division, print_function
import json
import os
from ansible.errors import AnsibleError
from ansible.plugins.lookup import LookupBase
from ansible.module_utils.urls import basic_auth_header, open_url
from ansible.module_utils._text import to_native
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.utils.display import Display
def grafana_switch_organisation(self, headers):
    try:
        r = open_url('%s/api/user/using/%s' % (self.grafana_url, self.grafana_org_id), headers=headers, method='POST')
    except HTTPError as e:
        raise GrafanaAPIException('Unable to switch to organization %s : %s' % (self.grafana_org_id, to_native(e)))
    if r.getcode() != 200:
        raise GrafanaAPIException('Unable to switch to organization %s : %s' % (self.grafana_org_id, str(r.getcode())))