from __future__ import (absolute_import, division, print_function)
import json
from ansible_collections.dellemc.openmanage.plugins.module_utils.redfish import Redfish, redfish_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import wait_for_job_completion, strip_substr_dict
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def lock_virtual_disk(module, redfish_obj):
    volume = module.params.get('volume_id')
    resp, job_uri, job_id = (None, None, None)
    controller_id = volume[0].split(':')[-1]
    check_id_exists(module, redfish_obj, 'controller_id', controller_id, CONTROLLER_URI)
    volume_uri = VOLUME_URI + '/{volume_id}'
    try:
        volume_resp = redfish_obj.invoke_request('GET', volume_uri.format(system_id=SYSTEM_ID, controller_id=controller_id, volume_id=volume[0]))
        links = volume_resp.json_data.get('Links')
        if links:
            for disk in volume_resp.json_data.get('Links').get('Drives'):
                drive_link = disk['@odata.id']
                drive_resp = redfish_obj.invoke_request('GET', drive_link)
                encryption_ability = drive_resp.json_data.get('EncryptionAbility')
                if encryption_ability != 'SelfEncryptingDrive':
                    module.fail_json(msg=PHYSICAL_DISK_ERR)
        lock_status = volume_resp.json_data.get('Oem').get('Dell').get('DellVolume').get('LockStatus')
    except HTTPError:
        module.fail_json(msg=PD_ERROR_MSG.format(controller_id))
    else:
        if lock_status == 'Unlocked' and module.check_mode:
            module.exit_json(msg=CHANGES_FOUND, changed=True)
        elif lock_status == 'Locked':
            module.exit_json(msg=NO_CHANGES_FOUND)
        else:
            resp = redfish_obj.invoke_request('POST', RAID_ACTION_URI.format(system_id=SYSTEM_ID, action='LockVirtualDisk'), data={'TargetFQDD': volume[0]})
            job_uri = resp.headers.get('Location')
            job_id = job_uri.split('/')[-1]
    return (resp, job_uri, job_id)