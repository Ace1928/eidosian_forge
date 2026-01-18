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
def filter_switch_controller_lldp_profile_data(json):
    option_list = ['tlvs_802dot1', 'tlvs_802dot3', 'auto_isl', 'auto_isl_auth', 'auto_isl_auth_encrypt', 'auto_isl_auth_identity', 'auto_isl_auth_macsec_profile', 'auto_isl_auth_reauth', 'auto_isl_auth_user', 'auto_isl_hello_timer', 'auto_isl_port_group', 'auto_isl_receive_timeout', 'auto_mclag_icl', 'custom_tlvs', 'med_location_service', 'med_network_policy', 'med_tlvs', 'name']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary