from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import open_url
def change_role_cardinality(module, auth, service_id, role, cardinality, force):
    data = {'cardinality': cardinality, 'force': force}
    try:
        status_result = open_url(auth.url + '/service/' + str(service_id) + '/role/' + role, method='PUT', force_basic_auth=True, url_username=auth.user, url_password=auth.password, data=module.jsonify(data))
    except Exception as e:
        module.fail_json(msg=str(e))
    if status_result.getcode() != 204:
        module.fail_json(msg='Failed to change cardinality for role: ' + role + '. Return code: ' + str(status_result.getcode()))