from __future__ import (absolute_import, division, print_function)
import os
import time
from ansible.module_utils.basic import AnsibleModule
def handle_elastigroup(client, module):
    has_changed = False
    group_id = None
    message = 'None'
    name = module.params.get('name')
    state = module.params.get('state')
    uniqueness_by = module.params.get('uniqueness_by')
    external_group_id = module.params.get('id')
    if uniqueness_by == 'id':
        if external_group_id is None:
            should_create = True
        else:
            should_create = False
            group_id = external_group_id
    else:
        groups = client.get_elastigroups()
        should_create, group_id = find_group_with_same_name(groups, name)
    if should_create is True:
        if state == 'present':
            eg = expand_elastigroup(module, is_update=False)
            module.debug(str(' [INFO] ' + message + '\n'))
            group = client.create_elastigroup(group=eg)
            group_id = group['id']
            message = 'Created group Successfully.'
            has_changed = True
        elif state == 'absent':
            message = 'Cannot delete non-existent group.'
            has_changed = False
    else:
        eg = expand_elastigroup(module, is_update=True)
        if state == 'present':
            group = client.update_elastigroup(group_update=eg, group_id=group_id)
            message = 'Updated group successfully.'
            try:
                roll_config = module.params.get('roll_config')
                if roll_config:
                    eg_roll = spotinst.aws_elastigroup.Roll(batch_size_percentage=roll_config.get('batch_size_percentage'), grace_period=roll_config.get('grace_period'), health_check_type=roll_config.get('health_check_type'))
                    roll_response = client.roll_group(group_roll=eg_roll, group_id=group_id)
                    message = 'Updated and started rolling the group successfully.'
            except SpotinstClientException as exc:
                message = 'Updated group successfully, but failed to perform roll. Error:' + str(exc)
            has_changed = True
        elif state == 'absent':
            try:
                client.delete_elastigroup(group_id=group_id)
            except SpotinstClientException as exc:
                if 'GROUP_DOESNT_EXIST' in exc.message:
                    pass
                else:
                    module.fail_json(msg='Error while attempting to delete group : ' + exc.message)
            message = 'Deleted group successfully.'
            has_changed = True
    return (group_id, message, has_changed)