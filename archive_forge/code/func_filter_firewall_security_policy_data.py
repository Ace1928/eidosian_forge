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
def filter_firewall_security_policy_data(json):
    option_list = ['action', 'app_category', 'app_group', 'application', 'application_list', 'av_profile', 'casb_profile', 'cifs_profile', 'comments', 'dlp_profile', 'dlp_sensor', 'dnsfilter_profile', 'dstaddr', 'dstaddr_negate', 'dstaddr4', 'dstaddr6', 'dstaddr6_negate', 'dstintf', 'emailfilter_profile', 'enforce_default_app_port', 'file_filter_profile', 'fsso_groups', 'global_label', 'groups', 'icap_profile', 'internet_service', 'internet_service_custom', 'internet_service_custom_group', 'internet_service_group', 'internet_service_id', 'internet_service_name', 'internet_service_negate', 'internet_service_src', 'internet_service_src_custom', 'internet_service_src_custom_group', 'internet_service_src_group', 'internet_service_src_id', 'internet_service_src_name', 'internet_service_src_negate', 'internet_service6', 'internet_service6_custom', 'internet_service6_custom_group', 'internet_service6_group', 'internet_service6_name', 'internet_service6_negate', 'internet_service6_src', 'internet_service6_src_custom', 'internet_service6_src_custom_group', 'internet_service6_src_group', 'internet_service6_src_name', 'internet_service6_src_negate', 'ips_sensor', 'ips_voip_filter', 'learning_mode', 'logtraffic', 'logtraffic_start', 'mms_profile', 'name', 'nat46', 'nat64', 'policyid', 'profile_group', 'profile_protocol_options', 'profile_type', 'schedule', 'sctp_filter_profile', 'send_deny_packet', 'service', 'service_negate', 'srcaddr', 'srcaddr_negate', 'srcaddr4', 'srcaddr6', 'srcaddr6_negate', 'srcintf', 'ssh_filter_profile', 'ssl_ssh_profile', 'status', 'url_category', 'users', 'utm_status', 'uuid', 'uuid_idx', 'videofilter_profile', 'virtual_patch_profile', 'voip_profile', 'webfilter_profile']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary