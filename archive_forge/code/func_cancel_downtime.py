from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def cancel_downtime(module, api_client):
    downtime = _get_downtime(module, api_client)
    api = DowntimesApi(api_client)
    if downtime is None:
        module.exit_json(changed=False)
    try:
        api.cancel_downtime(downtime['id'])
    except ApiException as e:
        module.fail_json(msg='Failed to create downtime: {0}'.format(e))
    module.exit_json(changed=True)