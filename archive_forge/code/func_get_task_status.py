from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
import time
import re
def get_task_status(self, task_id=None):
    """
        Monitor the status of a task in the Cisco Catalyst Center. It checks the task
        status periodically until the task is no longer 'In Progress'.
        If the task encounters an error or fails, it immediately fails the
        module and returns False.

        Parameters:
          - task_id: The ID of the task to monitor.

        Returns:
          - result: True if the task completed successfully, False otherwise.
        """
    result = False
    params = dict(task_id=task_id)
    while True:
        response = self.dnac_apply['exec'](family='task', function='get_task_by_id', params=params)
        response = response.response
        self.log('Task status for the task id {0} is {1}'.format(str(task_id), str(response)), 'INFO')
        if response.get('isError') or re.search('failed', response.get('progress'), flags=re.IGNORECASE):
            msg = 'Discovery task with id {0} has not completed - Reason: {1}'.format(task_id, response.get('failureReason'))
            self.log(msg, 'CRITICAL')
            self.module.fail_json(msg=msg)
            return False
        self.log('Task status for the task id (before checking status) {0} is {1}'.format(str(task_id), str(response)), 'INFO')
        progress = response.get('progress')
        if progress in ('In Progress', 'Inventory service initiating discovery'):
            time.sleep(3)
            continue
        else:
            result = True
            self.log('The Process is completed', 'INFO')
            break
    self.result.update(dict(discovery_task=response))
    return result