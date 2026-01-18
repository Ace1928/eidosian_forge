from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def firewall_proute(data, fos):
    vdom = data['vdom']
    firewall_proute_data = data['firewall_proute']
    filtered_data = underscore_to_hyphen(filter_firewall_proute_data(firewall_proute_data))
    converted_data = valid_attr_to_invalid_attrs(filtered_data)
    return fos.set('firewall', 'proute', data=converted_data, vdom=vdom)