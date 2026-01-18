from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils._text import to_native
from ansible.module_utils._text import to_text
from ansible_collections.community.grafana.plugins.module_utils.base import (
def grafana_organization_id_by_name(module, grafana_url, org_name, headers):
    r, info = fetch_url(module, '%s/api/user/orgs' % grafana_url, headers=headers, method='GET')
    if info['status'] != 200:
        raise GrafanaAPIException('Unable to retrieve users organizations: %s' % info)
    organizations = json.loads(to_text(r.read()))
    for org in organizations:
        if org['name'] == org_name:
            return org['orgId']
    raise GrafanaAPIException("Current user isn't member of organization: %s" % org_name)