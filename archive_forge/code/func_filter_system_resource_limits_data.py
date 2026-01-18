from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def filter_system_resource_limits_data(json):
    option_list = ['custom_service', 'dialup_tunnel', 'firewall_address', 'firewall_addrgrp', 'firewall_policy', 'ipsec_phase1', 'ipsec_phase1_interface', 'ipsec_phase2', 'ipsec_phase2_interface', 'log_disk_quota', 'onetime_schedule', 'proxy', 'recurring_schedule', 'service_group', 'session', 'sslvpn', 'user', 'user_group']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary