from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
from ansible.module_utils._text import to_native
@staticmethod
def get_differt_entries(list1, list2):
    """Return different entries of two lists"""
    return [a for a in list1 + list2 if a not in list1 or a not in list2]