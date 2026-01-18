from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
from ansible.module_utils._text import to_native
@staticmethod
def get_not_in_list_one(list1, list2):
    """Return entries that ore not in list one"""
    return [x for x in list1 if x not in set(list2)]