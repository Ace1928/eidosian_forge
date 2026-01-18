from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils._text import to_native
from ansible.module_utils.urls import open_url
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, wait_for_task, get_parent_datacenter
import os
def take_vm_screenshot(self):
    if self.current_vm_obj.runtime.powerState != vim.VirtualMachinePowerState.poweredOn:
        self.module.fail_json(msg='VM is %s, valid power state is poweredOn.' % self.current_vm_obj.runtime.powerState)
    try:
        task = self.current_vm_obj.CreateScreenshot_Task()
        wait_for_task(task)
    except vim.fault.FileFault as e:
        self.module.fail_json(msg='Failed to create screenshot due to errors when creating or accessing one or more files needed for this operation, %s' % to_native(e.msg))
    except vim.fault.InvalidState as e:
        self.module.fail_json(msg='Failed to create screenshot due to VM is not ready to respond to such requests, %s' % to_native(e.msg))
    except vmodl.RuntimeFault as e:
        self.module.fail_json(msg='Failed to create screenshot due to runtime fault, %s,' % to_native(e.msg))
    except vim.fault.TaskInProgress as e:
        self.module.fail_json(msg='Failed to create screenshot due to VM is busy, %s' % to_native(e.msg))
    if task.info.state == 'error':
        return {'changed': self.change_detected, 'failed': True, 'msg': task.info.error.msg}
    else:
        download_file_size = None
        self.change_detected = True
        file_url = self.generate_http_access_url(task.info.result)
        if self.params.get('local_path'):
            if file_url:
                download_file_size = self.download_screenshot_file(file_url=file_url, local_file_path=self.params['local_path'], file_name=task.info.result.split('/')[-1])
        screenshot_facts = self.get_screenshot_facts(task.info, file_url, download_file_size)
        return {'changed': self.change_detected, 'failed': False, 'screenshot_info': screenshot_facts}