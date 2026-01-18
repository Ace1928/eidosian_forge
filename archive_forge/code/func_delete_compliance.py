from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.compat.version import LooseVersion
def delete_compliance(module, rest_obj):
    """
    Deletes the list of baselines
    """
    valid_id_list = delete_idempotency_check(module, rest_obj)
    rest_obj.invoke_request('POST', DELETE_COMPLIANCE_BASELINE, data={'BaselineIds': valid_id_list})
    module.exit_json(msg=DELETE_MSG, changed=True)