from __future__ import (absolute_import, division, print_function)
import json
import os
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.common.dict_transformations import recursive_diff
def create_ad(module, rest_obj):
    prm = module.params
    if not prm.get('domain_server'):
        module.fail_json(msg=DOM_SERVER_MSG)
    if not prm.get('group_domain'):
        module.fail_json(msg=GRP_DOM_MSG)
    create_payload = make_payload(prm)
    msg = validate_n_testconnection(module, rest_obj, create_payload)
    if module.check_mode:
        module.exit_json(msg='{0}{1}'.format(msg, CHANGES_FOUND), changed=True)
    resp = rest_obj.invoke_request('POST', AD_URI, data=create_payload)
    ad = resp.json_data
    ad.pop('CertificateFile', '')
    module.exit_json(msg='{0}{1}'.format(msg, CREATE_SUCCESS), active_directory=ad, changed=True)