from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.urls import ConnectionError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def job_payload_for_update(rest_obj, module, target_data, baseline=None):
    """Formulate the payload to initiate a firmware update job."""
    resp = rest_obj.get_job_type_id('Update_Task')
    if resp is None:
        module.fail_json(msg='Unable to fetch the job type Id.')
    stage_dict = {'StageForNextReboot': 'true', 'RebootNow': 'false'}
    schedule = module.params['schedule']
    params = [{'Key': 'operationName', 'Value': 'INSTALL_FIRMWARE'}, {'Key': 'stagingValue', 'Value': stage_dict[schedule]}, {'Key': 'signVerify', 'Value': 'true'}]
    if schedule == 'RebootNow':
        reboot_dict = {'PowerCycle': '1', 'GracefulReboot': '2', 'GracefulRebootForce': '3'}
        reboot_type = module.params['reboot_type']
        params.append({'Key': 'rebootType', 'Value': reboot_dict[reboot_type]})
    payload = {'Id': 0, 'JobName': 'Firmware Update Task', 'JobDescription': FW_JOB_DESC, 'Schedule': 'startnow', 'State': 'Enabled', 'JobType': {'Id': resp, 'Name': 'Update_Task'}, 'Targets': target_data, 'Params': params}
    if baseline is not None:
        payload['Params'].append({'Key': 'complianceReportId', 'Value': '{0}'.format(baseline['baseline_id'])})
        payload['Params'].append({'Key': 'repositoryId', 'Value': '{0}'.format(baseline['repo_id'])})
        payload['Params'].append({'Key': 'catalogId', 'Value': '{0}'.format(baseline['catalog_id'])})
        payload['Params'].append({'Key': 'complianceUpdate', 'Value': 'true'})
    else:
        payload['Params'].append({'JobId': 0, 'Key': 'complianceUpdate', 'Value': 'false'})
    return payload