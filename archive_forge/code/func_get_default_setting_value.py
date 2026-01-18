from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, option_diff, vmware_argument_spec
from ansible.module_utils._text import to_native
def get_default_setting_value(self, setting_key):
    return self.option_manager.QueryOptions(name=setting_key)[0].value