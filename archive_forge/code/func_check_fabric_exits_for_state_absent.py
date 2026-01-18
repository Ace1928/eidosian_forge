from __future__ import (absolute_import, division, print_function)
import json
import socket
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ssl import SSLError
def check_fabric_exits_for_state_absent(fabric_values, module, fabric_name):
    """
    idempotency check in case of state absent
    :param fabric_values: fabric details of existing fabric
    :param module: ansible module object
    :param fabric_name: fabric name
    :return: str -  fabric id
    """
    fabric_id, fabric_details = get_fabric_id_details(fabric_name, fabric_values)
    if module.check_mode and fabric_id is None:
        module.exit_json(msg=CHECK_MODE_CHANGE_NOT_FOUND_MSG)
    if module.check_mode and fabric_id is not None:
        module.exit_json(msg=CHECK_MODE_CHANGE_FOUND_MSG, changed=True)
    if not module.check_mode and fabric_id is None:
        module.exit_json(msg=FABRIC_NOT_FOUND_ERROR_MSG.format(fabric_name))
    return fabric_id