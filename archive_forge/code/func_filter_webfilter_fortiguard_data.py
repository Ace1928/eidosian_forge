from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def filter_webfilter_fortiguard_data(json):
    option_list = ['cache_mem_percent', 'cache_mem_permille', 'cache_mode', 'cache_prefix_match', 'close_ports', 'embed_image', 'ovrd_auth_https', 'ovrd_auth_port', 'ovrd_auth_port_http', 'ovrd_auth_port_https', 'ovrd_auth_port_https_flow', 'ovrd_auth_port_warning', 'request_packet_size_limit', 'warn_auth_https']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary