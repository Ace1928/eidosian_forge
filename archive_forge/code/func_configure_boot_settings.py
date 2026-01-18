from __future__ import (absolute_import, division, print_function)
import json
import time
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI, idrac_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import (strip_substr_dict, idrac_system_reset,
from ansible.module_utils.basic import AnsibleModule
def configure_boot_settings(module, idrac, res_id):
    job_resp, payload = ({}, {'Boot': {}})
    boot_order = module.params.get('boot_order')
    override_mode = module.params.get('boot_source_override_mode')
    override_enabled = module.params.get('boot_source_override_enabled')
    override_target = module.params.get('boot_source_override_target')
    response = get_response_attributes(module, idrac, res_id)
    if boot_order is not None:
        exist_boot_order = response.get('BootOrder')
        invalid_boot_order = [bo for bo in boot_order if bo not in exist_boot_order]
        if invalid_boot_order:
            module.fail_json(msg=INVALID_BOOT_OPT.format('Invalid'), invalid_boot_order=invalid_boot_order)
        if not len(set(boot_order)) == len(boot_order):
            dup_order = boot_order[:]
            [dup_order.remove(bo) for bo in exist_boot_order if bo in dup_order]
            module.fail_json(msg=INVALID_BOOT_OPT.format('Duplicate'), duplicate_boot_order=dup_order)
        if not len(boot_order) == len(exist_boot_order):
            module.fail_json(msg='Unable to complete the operation because all boot devices are required for this operation.')
        if not boot_order == exist_boot_order:
            payload['Boot'].update({'BootOrder': boot_order})
    if override_mode is not None and (not BS_OVERRIDE_MODE.get(override_mode) == response.get('BootSourceOverrideMode')):
        payload['Boot'].update({'BootSourceOverrideMode': BS_OVERRIDE_MODE.get(override_mode)})
    if override_enabled is not None and (not BS_OVERRIDE_ENABLED.get(override_enabled) == response.get('BootSourceOverrideEnabled')):
        payload['Boot'].update({'BootSourceOverrideEnabled': BS_OVERRIDE_ENABLED.get(override_enabled)})
    if override_target is not None and (not BS_OVERRIDE_TARGET.get(override_target) == response.get('BootSourceOverrideTarget')):
        payload['Boot'].update({'BootSourceOverrideTarget': BS_OVERRIDE_TARGET.get(override_target)})
        uefi_override_target = module.params.get('uefi_target_boot_source_override')
        if override_target == 'uefi_target' and (not uefi_override_target == response.get('UefiTargetBootSourceOverride')):
            payload['Boot'].update({'UefiTargetBootSourceOverride': uefi_override_target})
    if module.check_mode and payload['Boot']:
        module.exit_json(msg=CHANGES_MSG, changed=True)
    elif (module.check_mode or not module.check_mode) and (not payload['Boot']):
        module.exit_json(msg=NO_CHANGES_MSG)
    else:
        job_resp = apply_boot_settings(module, idrac, payload, res_id)
    return job_resp