from __future__ import (absolute_import, division, print_function)
import json
import copy
from ssl import SSLError
from ansible_collections.dellemc.openmanage.plugins.module_utils.redfish import Redfish, redfish_auth_params
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import MANAGER_JOB_ID_URI, wait_for_redfish_reboot_job, \
def configure_raid_operation(module, session_obj):
    """
    configure raid action based on state and command input
    """
    module_params = module.params
    state = module_params.get('state')
    command = module_params.get('command')
    if state is not None and state == 'present':
        return perform_volume_create_modify(module, session_obj)
    elif state is not None and state == 'absent':
        return perform_volume_deletion(module, session_obj)
    elif command is not None and command == 'initialize':
        return perform_volume_initialization(module, session_obj)