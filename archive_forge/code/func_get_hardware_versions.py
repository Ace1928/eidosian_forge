from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import find_obj, vmware_argument_spec, PyVmomi
from ansible_collections.community.vmware.plugins.module_utils.vm_device_helper import PyVmomiDeviceHelper
def get_hardware_versions(self, env_browser):
    support_create = []
    default_config = ''
    try:
        desc = env_browser.QueryConfigOptionDescriptor()
    except Exception as e:
        self.module.fail_json(msg='Failed to obtain VM config option descriptor due to fault: %s' % to_native(e))
    if desc:
        for option_desc in desc:
            if option_desc.createSupported:
                support_create = support_create + [option_desc.key]
            if option_desc.defaultConfigOption:
                default_config = option_desc.key
    return (support_create, default_config)