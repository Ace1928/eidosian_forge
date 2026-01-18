from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.common.dict_transformations import recursive_diff
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import get_item_and_list
def delete_uplink(module, rest_obj, fabric_id, uplink_id):
    if module.check_mode:
        module.exit_json(changed=True, msg=CHECK_MODE_MSG)
    rest_obj.invoke_request('DELETE', UPLINK_URI.format(fabric_id=fabric_id, uplink_id=uplink_id))
    module.exit_json(msg='Successfully deleted the uplink.', changed=True)