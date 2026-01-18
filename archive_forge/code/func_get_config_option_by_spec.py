from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import find_obj, vmware_argument_spec, PyVmomi
from ansible_collections.community.vmware.plugins.module_utils.vm_device_helper import PyVmomiDeviceHelper
def get_config_option_by_spec(self, env_browser, guest_id=None, key=''):
    vm_config_option = None
    if guest_id is None:
        guest_id = []
    if self.is_vcenter():
        host = self.target_host
    else:
        host = None
    config_query_spec = vim.EnvironmentBrowser.ConfigOptionQuerySpec(guestId=guest_id, host=host, key=key)
    try:
        vm_config_option = env_browser.QueryConfigOptionEx(spec=config_query_spec)
    except Exception as e:
        self.module.fail_json(msg='Failed to obtain VM config options due to fault: %s' % to_native(e))
    return vm_config_option