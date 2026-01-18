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
def filter_firewall_consolidated_policy_data(json):
    option_list = ['action', 'application_list', 'auto_asic_offload', 'av_profile', 'captive_portal_exempt', 'cifs_profile', 'comments', 'diffserv_forward', 'diffserv_reverse', 'diffservcode_forward', 'diffservcode_rev', 'dlp_sensor', 'dnsfilter_profile', 'dstaddr_negate', 'dstaddr4', 'dstaddr6', 'dstintf', 'emailfilter_profile', 'fixedport', 'fsso_groups', 'global_label', 'groups', 'http_policy_redirect', 'icap_profile', 'inbound', 'inspection_mode', 'internet_service', 'internet_service_custom', 'internet_service_custom_group', 'internet_service_group', 'internet_service_id', 'internet_service_negate', 'internet_service_src', 'internet_service_src_custom', 'internet_service_src_custom_group', 'internet_service_src_group', 'internet_service_src_id', 'internet_service_src_negate', 'ippool', 'ips_sensor', 'logtraffic', 'logtraffic_start', 'mms_profile', 'name', 'nat', 'outbound', 'per_ip_shaper', 'policyid', 'poolname4', 'poolname6', 'profile_group', 'profile_protocol_options', 'profile_type', 'schedule', 'service', 'service_negate', 'session_ttl', 'srcaddr_negate', 'srcaddr4', 'srcaddr6', 'srcintf', 'ssh_filter_profile', 'ssh_policy_redirect', 'ssl_ssh_profile', 'status', 'tcp_mss_receiver', 'tcp_mss_sender', 'traffic_shaper', 'traffic_shaper_reverse', 'users', 'utm_status', 'uuid', 'voip_profile', 'vpntunnel', 'waf_profile', 'wanopt', 'wanopt_detection', 'wanopt_passive_opt', 'wanopt_peer', 'wanopt_profile', 'webcache', 'webcache_https', 'webfilter_profile', 'webproxy_forward_server', 'webproxy_profile']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary