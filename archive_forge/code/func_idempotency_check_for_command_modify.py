from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.compat.version import LooseVersion
def idempotency_check_for_command_modify(current_payload, expected_payload, module):
    """
    idempotency check in case of modify operation
    :param current_payload: payload modify
    :param expected_payload: already existing payload for specified.
    :param module: ansible module object
    :return: None
    """
    payload_diff = compare_payloads(expected_payload, current_payload)
    if module.check_mode:
        if payload_diff:
            module.exit_json(msg=CHECK_MODE_CHANGES_MSG, changed=True)
        else:
            module.exit_json(msg=CHECK_MODE_NO_CHANGES_MSG, changed=False)
    elif not module.check_mode and (not payload_diff):
        module.exit_json(msg=IDEMPOTENCY_MSG, changed=False)