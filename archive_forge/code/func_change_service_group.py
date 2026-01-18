from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import open_url
def change_service_group(module, auth, service_id, group_id):
    data = {'action': {'perform': 'chgrp', 'params': {'group_id': group_id}}}
    try:
        status_result = open_url(auth.url + '/service/' + str(service_id) + '/action', method='POST', force_basic_auth=True, url_username=auth.user, url_password=auth.password, data=module.jsonify(data))
    except Exception as e:
        module.fail_json(msg=str(e))