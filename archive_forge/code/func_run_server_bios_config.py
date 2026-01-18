from __future__ import (absolute_import, division, print_function)
import json
import time
from ansible.module_utils.common.dict_transformations import recursive_diff
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.dellemc_idrac import iDRACConnection, idrac_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import idrac_redfish_job_tracking, \
def run_server_bios_config(idrac, module):
    msg = {}
    idrac.use_redfish = True
    _validate_params(module.params['boot_sources'])
    if module.check_mode:
        idrac.config_mgr.is_change_applicable()
    msg = idrac.config_mgr.configure_boot_sources(input_boot_devices=module.params['boot_sources'])
    return msg