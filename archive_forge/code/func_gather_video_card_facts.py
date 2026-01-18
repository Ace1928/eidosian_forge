from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, wait_for_task
def gather_video_card_facts(self, vm_obj):
    """
        Gather facts about VM's video card settings
        Args:
            vm_obj: Managed object of virtual machine
        Returns: Video Card device and a list of dict video card configuration
        """
    video_facts = dict()
    video_card = None
    for device in vm_obj.config.hardware.device:
        if isinstance(device, vim.vm.device.VirtualVideoCard):
            video_card = device
            video_facts = dict(auto_detect=device.useAutoDetect, display_number=device.numDisplays, video_memory=device.videoRamSizeInKB, enable_3D_support=device.enable3DSupport, renderer_3D=device.use3dRenderer, memory_3D=device.graphicsMemorySizeInKB)
            break
    return (video_card, video_facts)