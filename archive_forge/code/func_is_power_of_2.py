from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, wait_for_task
@staticmethod
def is_power_of_2(num):
    return num != 0 and num & num - 1 == 0