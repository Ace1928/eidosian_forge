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
def filter_firewall_interface_policy_data(json):
    option_list = ['address_type', 'application_list', 'application_list_status', 'av_profile', 'av_profile_status', 'casb_profile', 'casb_profile_status', 'comments', 'dlp_profile', 'dlp_profile_status', 'dlp_sensor', 'dlp_sensor_status', 'dsri', 'dstaddr', 'emailfilter_profile', 'emailfilter_profile_status', 'interface', 'ips_sensor', 'ips_sensor_status', 'label', 'logtraffic', 'policyid', 'scan_botnet_connections', 'service', 'spamfilter_profile', 'spamfilter_profile_status', 'srcaddr', 'status', 'webfilter_profile', 'webfilter_profile_status']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary