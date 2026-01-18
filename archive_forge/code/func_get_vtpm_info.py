from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi, wait_for_task
from ansible_collections.community.vmware.plugins.module_utils.vm_device_helper import PyVmomiDeviceHelper
def get_vtpm_info(self, vm_obj=None, vtpm_device=None):
    vtpm_info = dict()
    if vm_obj:
        for device in vm_obj.config.hardware.device:
            if self.device_helper.is_tpm_device(device):
                vtpm_device = device
    if vtpm_device:
        vtpm_info = dict(key=vtpm_device.key, label=vtpm_device.deviceInfo.label, summary=vtpm_device.deviceInfo.summary)
    return vtpm_info