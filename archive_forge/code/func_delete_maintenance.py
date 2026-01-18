from __future__ import absolute_import, division, print_function
import datetime
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.urls import open_url
def delete_maintenance(auth_headers, url, statuspage, maintenance_id):
    try:
        values = json.dumps({'statuspage_id': statuspage, 'maintenance_id': maintenance_id})
        response = open_url(url=url + '/v2/maintenance/delete', data=values, headers=auth_headers)
        data = json.loads(response.read())
        if data['status']['error'] == 'yes':
            return (1, None, 'Invalid maintenance_id')
    except Exception as e:
        return (1, None, to_native(e))
    return (0, None, None)