from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_text
from ansible_collections.community.grafana.plugins.module_utils.base import (
from ansible.module_utils.urls import basic_auth_header
def grafana_delete_notification_channel(self, data):
    r, info = fetch_url(self._module, '%s/api/alert-notifications/uid/%s' % (data['url'], data['uid']), headers=self.headers, method='DELETE')
    if info['status'] == 200:
        return {'state': 'absent', 'changed': True}
    elif info['status'] == 404:
        return {'changed': False}
    else:
        raise GrafanaAPIException('Unable to delete notification channel %s : %s' % (data['uid'], info))