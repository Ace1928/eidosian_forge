from __future__ import absolute_import, division, print_function
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.rest_application import RestApplication
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_volume
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_lun_details(self, lun):
    """
        Extract LUN details, from XML to python dict

        :return: Details about the lun
        :rtype: dict
        """
    if self.use_rest:
        return lun
    return_value = {'size': int(lun.get_child_content('size'))}
    bool_attr_map = {'is-space-alloc-enabled': 'space_allocation', 'is-space-reservation-enabled': 'space_reserve'}
    for attr in bool_attr_map:
        value = lun.get_child_content(attr)
        if value is not None:
            return_value[bool_attr_map[attr]] = self.na_helper.get_value_for_bool(True, value)
    str_attr_map = {'comment': 'comment', 'multiprotocol-type': 'os_type', 'name': 'name', 'path': 'path', 'qos-policy-group': 'qos_policy_group', 'qos-adaptive-policy-group': 'qos_adaptive_policy_group'}
    for attr in str_attr_map:
        value = lun.get_child_content(attr)
        if value is None and attr in ('comment', 'qos-policy-group', 'qos-adaptive-policy-group'):
            value = ''
        if value is not None:
            return_value[str_attr_map[attr]] = value
    return return_value