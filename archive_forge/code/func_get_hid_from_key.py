from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
def get_hid_from_key(self, key):
    if key == ' ':
        return ('0x2c', [])
    for keys_name, key_code, keys_value in self.keys_hid_code:
        if isinstance(keys_name, tuple):
            for keys in keys_value:
                if key == keys[0]:
                    return (key_code, keys[1])
        elif key == keys_name:
            return (key_code, keys_value[0][1])