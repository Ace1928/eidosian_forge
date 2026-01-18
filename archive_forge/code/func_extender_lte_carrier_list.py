from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def extender_lte_carrier_list(data, fos):
    vdom = data['vdom']
    extender_lte_carrier_list_data = data['extender_lte_carrier_list']
    filtered_data = underscore_to_hyphen(filter_extender_lte_carrier_list_data(extender_lte_carrier_list_data))
    converted_data = valid_attr_to_invalid_attrs(filtered_data)
    return fos.set('extender', 'lte-carrier-list', data=converted_data, vdom=vdom)