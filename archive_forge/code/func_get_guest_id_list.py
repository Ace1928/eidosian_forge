from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import find_obj, vmware_argument_spec, PyVmomi
from ansible_collections.community.vmware.plugins.module_utils.vm_device_helper import PyVmomiDeviceHelper
def get_guest_id_list(self, guest_os_desc):
    gos_id_list = []
    if guest_os_desc:
        for gos_desc in guest_os_desc.guestOSDescriptor:
            gos_id_list = gos_id_list + [gos_desc.id]
    return gos_id_list