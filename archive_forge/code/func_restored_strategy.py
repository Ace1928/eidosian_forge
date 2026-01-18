from __future__ import absolute_import, division, print_function
import datetime
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.scaleway import (
def restored_strategy(module, account_api, backup):
    if backup is None:
        module.fail_json(msg='Backup "%s" not found' % module.params['id'])
    database_name = module.params['database_name']
    instance_id = module.params['instance_id']
    if module.check_mode:
        module.exit_json(changed=True)
    backup = wait_to_complete_state_transition(module, account_api, backup)
    payload = {'database_name': database_name, 'instance_id': instance_id}
    response = account_api.post('/rdb/v1/regions/%s/backups/%s/restore' % (module.params.get('region'), backup['id']), payload)
    if response.ok:
        result = wait_to_complete_state_transition(module, account_api, response.json)
        module.exit_json(changed=True, metadata=result)
    module.fail_json(msg='Error restoring backup [{0}: {1}]'.format(response.status_code, response.json))