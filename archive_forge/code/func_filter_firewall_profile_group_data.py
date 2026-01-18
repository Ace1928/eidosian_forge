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
def filter_firewall_profile_group_data(json):
    option_list = ['application_list', 'av_profile', 'casb_profile', 'cifs_profile', 'dlp_profile', 'dlp_sensor', 'dnsfilter_profile', 'emailfilter_profile', 'file_filter_profile', 'icap_profile', 'ips_sensor', 'ips_voip_filter', 'mms_profile', 'name', 'profile_protocol_options', 'sctp_filter_profile', 'spamfilter_profile', 'ssh_filter_profile', 'ssl_ssh_profile', 'videofilter_profile', 'virtual_patch_profile', 'voip_profile', 'waf_profile', 'webfilter_profile']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary