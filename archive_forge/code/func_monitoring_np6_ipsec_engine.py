from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def monitoring_np6_ipsec_engine(data, fos):
    vdom = data['vdom']
    monitoring_np6_ipsec_engine_data = data['monitoring_np6_ipsec_engine']
    monitoring_np6_ipsec_engine_data = flatten_multilists_attributes(monitoring_np6_ipsec_engine_data)
    filtered_data = underscore_to_hyphen(filter_monitoring_np6_ipsec_engine_data(monitoring_np6_ipsec_engine_data))
    return fos.set('monitoring', 'np6-ipsec-engine', data=filtered_data, vdom=vdom)