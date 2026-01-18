from __future__ import (absolute_import, division, print_function)
import json
from ansible_collections.dellemc.openmanage.plugins.module_utils.redfish import Redfish, redfish_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import wait_for_job_completion, strip_substr_dict
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def check_id_exists(module, redfish_obj, key, item_id, uri):
    msg = "{0} with id '{1}' not found in system".format(key, item_id)
    try:
        resp = redfish_obj.invoke_request('GET', uri.format(system_id=SYSTEM_ID, controller_id=item_id))
        if not resp.success:
            module.exit_json(msg=msg, failed=True)
    except HTTPError as err:
        module.exit_json(msg=msg, error_info=json.load(err), failed=True)