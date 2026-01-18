from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def filter_endpoint_control_fctems_override_data(json):
    option_list = ['call_timeout', 'capabilities', 'cloud_server_type', 'dirty_reason', 'ems_id', 'fortinetone_cloud_authentication', 'https_port', 'interface', 'interface_select_method', 'name', 'out_of_sync_threshold', 'preserve_ssl_session', 'pull_avatars', 'pull_malware_hash', 'pull_sysinfo', 'pull_tags', 'pull_vulnerabilities', 'serial_number', 'server', 'source_ip', 'status', 'tenant_id', 'trust_ca_cn', 'websocket_override']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary