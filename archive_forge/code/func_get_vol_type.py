from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
import copy
def get_vol_type(vol_type):
    """
    :param vol_type: Type of the volume
    :return: Corresponding value for the entered vol_type
    """
    vol_type_dict = {'THICK_PROVISIONED': 'ThickProvisioned', 'THIN_PROVISIONED': 'ThinProvisioned'}
    return vol_type_dict.get(vol_type)