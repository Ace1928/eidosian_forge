from __future__ import absolute_import, division, print_function
import re
import time
from ansible.module_utils.basic import AnsibleModule
def configlet_update_task(module):
    """ Poll device info of switch from CVP up to three times to see if the
        configlet updates have spawned a task. It sometimes takes a second for
        the task to be spawned after configlet updates. If a task is found
        return the task ID. Otherwise return None.

    :param module: Ansible module with parameters and client connection.
    :return: Task ID or None.
    """
    for num in range(3):
        device_info = switch_info(module)
        if 'taskIdList' in device_info and len(device_info['taskIdList']) > 0:
            for task in device_info['taskIdList']:
                if 'Configlet Assign' in task['description'] and task['data']['WORKFLOW_ACTION'] == 'Configlet Push':
                    return task['workOrderId']
        time.sleep(1)
    return None