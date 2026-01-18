from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.comparison import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.comparison import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.comparison import (
def filter_system_link_monitor_data(json):
    option_list = ['addr_mode', 'class_id', 'diffservcode', 'fail_weight', 'failtime', 'gateway_ip', 'gateway_ip6', 'ha_priority', 'http_agent', 'http_get', 'http_match', 'interval', 'name', 'packet_size', 'password', 'port', 'probe_count', 'probe_timeout', 'protocol', 'recoverytime', 'route', 'security_mode', 'server', 'server_config', 'server_list', 'server_type', 'service_detection', 'source_ip', 'source_ip6', 'srcintf', 'status', 'update_cascade_interface', 'update_policy_route', 'update_static_route']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary