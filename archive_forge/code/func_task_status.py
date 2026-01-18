from __future__ import absolute_import, division, print_function
import os
import time
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.general.plugins.module_utils.proxmox import (proxmox_auth_argument_spec, ProxmoxAnsible)
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def task_status(self, node, taskid, timeout):
    """
        Check the task status and wait until the task is completed or the timeout is reached.
        """
    while timeout:
        if self.api_task_ok(node, taskid):
            return True
        timeout = timeout - 1
        if timeout == 0:
            self.module.fail_json(msg='Reached timeout while waiting for uploading/downloading template. Last line in task before timeout: %s' % self.proxmox_api.node(node).tasks(taskid).log.get()[:1])
        time.sleep(1)
    return False