from __future__ import absolute_import, division, print_function
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell import (
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.powerflex_base \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
from ansible.module_utils.basic import AnsibleModule
def is_rename_required(self, fault_set_details, fault_set_params):
    """To get the details of the fields to be modified."""
    if fault_set_params['fault_set_new_name'] is not None and fault_set_params['fault_set_new_name'] != fault_set_details['name']:
        return True
    return False