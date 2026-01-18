from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import apply_diff_key, job_tracking
def get_deploy_payload(module_params, deviceidlist, template_id):
    deploy_payload = {}
    if isinstance(module_params.get('attributes'), dict):
        deploy_payload.update(module_params.get('attributes'))
    deploy_payload['Id'] = template_id
    deploy_payload['TargetIds'] = deviceidlist
    return deploy_payload