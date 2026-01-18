from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils._text import to_native
from ansible.module_utils.urls import open_url
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, wait_for_task, get_parent_datacenter
import os
def get_screenshot_facts(self, task_info, file_url, file_size):
    screenshot_facts = dict()
    if task_info is not None:
        screenshot_facts = dict(virtual_machine=task_info.entityName, screenshot_file=task_info.result, task_start_time=task_info.startTime, task_complete_time=task_info.completeTime, result=task_info.state, screenshot_file_url=file_url, download_local_path=self.params.get('local_path'), download_file_size=file_size)
    return screenshot_facts