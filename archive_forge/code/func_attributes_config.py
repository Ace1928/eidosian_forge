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
def attributes_config(module, redfish_obj):
    curr_resp = get_current_attributes(redfish_obj)
    curr_attr = curr_resp.get('Attributes', {})
    inp_attr = module.params.get('attributes')
    diff_tuple = recursive_diff(inp_attr, curr_attr)
    attr = {}
    if diff_tuple:
        if diff_tuple[0]:
            attr = diff_tuple[0]
    invalid = {}
    attr_registry = get_attributes_registry(redfish_obj)
    if attr_registry:
        invalid.update(validate_vs_registry(attr_registry, attr))
        if invalid:
            module.exit_json(failed=True, status_msg=INVALID_ATTRIBUTES_MSG, invalid_attributes=invalid)
    if not attr:
        module.exit_json(status_msg=NO_CHANGES_MSG)
    if module.check_mode:
        module.exit_json(status_msg=CHANGES_MSG, changed=True)
    pending = get_pending_attributes(redfish_obj)
    pending.update(attr)
    if pending:
        job_id, job_state = check_scheduled_bios_job(redfish_obj)
        if job_id:
            if job_state in ['Running', 'Starting']:
                module.exit_json(status_msg=BIOS_JOB_RUNNING, job_id=job_id, failed=True)
            elif job_state in ['Scheduled', 'Scheduling']:
                delete_scheduled_bios_job(redfish_obj, job_id)
    rf_settings = curr_resp.get('@Redfish.Settings', {}).get('SupportedApplyTimes', [])
    job_id, reboot_required = apply_attributes(module, redfish_obj, pending, rf_settings)
    if reboot_required and job_id:
        reset_success = reset_host(module, redfish_obj)
        if not reset_success:
            module.exit_json(status_msg='Attributes committed but reboot has failed {0}'.format(HOST_RESTART_FAILED), failed=True)
        if module.params.get('job_wait'):
            job_failed, msg, job_dict, wait_time = idrac_redfish_job_tracking(redfish_obj, iDRAC_JOB_URI.format(job_id=job_id), max_job_wait_sec=module.params.get('job_wait_timeout'))
            if job_failed:
                module.exit_json(failed=True, status_msg=msg, job_id=job_id)
            module.exit_json(status_msg=SUCCESS_COMPLETE, job_id=job_id, msg=strip_substr_dict(job_dict), changed=True)
        else:
            module.exit_json(status_msg=SCHEDULED_SUCCESS, job_id=job_id, changed=True)
    module.exit_json(status_msg=COMMITTED_SUCCESS.format(module.params.get('apply_time')), job_id=job_id, changed=True)