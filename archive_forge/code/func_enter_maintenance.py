from __future__ import absolute_import, division, print_function
import os
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, get_all_objs, wait_for_task, PyVmomi
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils._text import to_native
def enter_maintenance(self):
    try:
        task = self.host.EnterMaintenanceMode_Task(timeout=15)
        success, result = wait_for_task(task)
    except Exception as e:
        self.module.fail_json(msg='Failed to enter maintenance mode. Ensure that there are no powered on machines on the host. %s' % to_native(e))