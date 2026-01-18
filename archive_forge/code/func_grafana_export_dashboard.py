from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils._text import to_native
from ansible.module_utils._text import to_text
from ansible_collections.community.grafana.plugins.module_utils.base import (
def grafana_export_dashboard(module, data):
    headers = grafana_headers(module, data)
    grafana_version = get_grafana_version(module, data['url'], headers)
    if grafana_version < 5:
        if data.get('slug'):
            uid = data['slug']
        else:
            raise GrafanaMalformedJson('No slug parameter. Needed with grafana < 5')
    elif data.get('uid'):
        uid = data['uid']
    else:
        raise GrafanaExportException('No uid specified')
    dashboard_exists, dashboard = grafana_dashboard_exists(module, data['url'], uid, headers=headers)
    if dashboard_exists is True:
        if module.check_mode:
            module.exit_json(uid=uid, failed=False, changed=True, msg='Dashboard %s will be exported to %s' % (uid, data['path']))
        try:
            with open(data['path'], 'w', encoding='utf-8') as f:
                f.write(json.dumps(dashboard, indent=2))
        except Exception as e:
            raise GrafanaExportException("Can't write json file : %s" % to_native(e))
        result = {'msg': 'Dashboard %s exported to %s' % (uid, data['path']), 'uid': uid, 'changed': True}
    else:
        result = {'msg': 'Dashboard %s does not exist.' % uid, 'uid': uid, 'changed': False}
    return result