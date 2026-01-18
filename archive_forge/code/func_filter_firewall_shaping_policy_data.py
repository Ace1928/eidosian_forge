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
def filter_firewall_shaping_policy_data(json):
    option_list = ['app_category', 'app_group', 'application', 'class_id', 'comment', 'cos', 'cos_mask', 'diffserv_forward', 'diffserv_reverse', 'diffservcode_forward', 'diffservcode_rev', 'dstaddr', 'dstaddr6', 'dstintf', 'groups', 'id', 'internet_service', 'internet_service_custom', 'internet_service_custom_group', 'internet_service_group', 'internet_service_id', 'internet_service_name', 'internet_service_src', 'internet_service_src_custom', 'internet_service_src_custom_group', 'internet_service_src_group', 'internet_service_src_id', 'internet_service_src_name', 'ip_version', 'name', 'per_ip_shaper', 'schedule', 'service', 'srcaddr', 'srcaddr6', 'srcintf', 'status', 'tos', 'tos_mask', 'tos_negate', 'traffic_shaper', 'traffic_shaper_reverse', 'traffic_type', 'url_category', 'users', 'uuid']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary