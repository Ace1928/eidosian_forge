from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def require_modify(desired, current):
    for condition in desired:
        if condition.get('operator'):
            for current_condition in current:
                if condition['operator'] == current_condition['operator']:
                    condition_modified = self.na_helper.get_modified_attributes(current_condition, condition)
                    if condition_modified:
                        return True
        else:
            return True