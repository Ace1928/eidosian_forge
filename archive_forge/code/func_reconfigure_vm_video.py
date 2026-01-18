from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, wait_for_task
def reconfigure_vm_video(self, vm_obj):
    """
        Reconfigure video card settings of virtual machine
        Args:
            vm_obj: Managed object of virtual machine
        Returns: Reconfigure results
        """
    video_card_spec = self.get_video_card_spec(vm_obj)
    if video_card_spec is None:
        return {'changed': False, 'failed': False, 'instance': self.video_card_facts}
    self.config_spec.deviceChange.append(video_card_spec)
    try:
        task = vm_obj.ReconfigVM_Task(spec=self.config_spec)
        wait_for_task(task)
    except vim.fault.InvalidDeviceSpec as invalid_device_spec:
        self.module.fail_json(msg='Failed to configure video card on given virtual machine due to invalid device spec : %s' % to_native(invalid_device_spec.msg), details='Please check ESXi server logs for more details.')
    except vim.fault.RestrictedVersion as e:
        self.module.fail_json(msg='Failed to reconfigure virtual machine due to product versioning restrictions: %s' % to_native(e.msg))
    if task.info.state == 'error':
        return {'changed': self.change_detected, 'failed': True, 'msg': task.info.error.msg}
    video_card_facts = self.gather_video_card_facts(vm_obj)[1]
    return {'changed': self.change_detected, 'failed': False, 'instance': video_card_facts}