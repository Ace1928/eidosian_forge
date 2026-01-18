from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
import copy
def get_access_mode(access_mode):
    """
    :param access_mode: Access mode of the SDC
    :return: The enum for the access mode
    """
    access_mode_dict = {'READ_WRITE': 'ReadWrite', 'READ_ONLY': 'ReadOnly', 'NO_ACCESS': 'NoAccess'}
    return access_mode_dict.get(access_mode)