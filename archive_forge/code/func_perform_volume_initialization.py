from __future__ import (absolute_import, division, print_function)
import json
import copy
from ssl import SSLError
from ansible_collections.dellemc.openmanage.plugins.module_utils.redfish import Redfish, redfish_auth_params
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import MANAGER_JOB_ID_URI, wait_for_redfish_reboot_job, \
def perform_volume_initialization(module, session_obj):
    """
    perform volume initialization for command initialize
    """
    specified_volume_id = module.params.get('volume_id')
    if specified_volume_id:
        operations = check_initialization_progress(module, session_obj, specified_volume_id)
        if operations:
            operation_message = 'Cannot perform the configuration operations because a configuration job for the device already exists.'
            operation_name = operations[0].get('OperationName')
            percentage_complete = operations[0].get('PercentageComplete')
            if operation_name and percentage_complete:
                operation_message = "Cannot perform the configuration operation because the configuration job '{0}' in progress is at '{1}' percentage.".format(operation_name, percentage_complete)
            module.fail_json(msg=operation_message)
        else:
            method = 'POST'
            uri = VOLUME_INITIALIZE_URI.format(storage_base_uri=storage_collection_map['storage_base_uri'], volume_id=specified_volume_id)
            payload = {'InitializeType': module.params['initialize_type']}
            return perform_storage_volume_action(method, uri, session_obj, 'initialize', payload)
    else:
        module.fail_json(msg="'volume_id' option is a required property for initializing a volume.")