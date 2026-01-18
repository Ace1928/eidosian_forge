from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def is_none_or_empty_string(param):
    """ validates the input string for None or empty values
    """
    return not param or len(str(param)) <= 0