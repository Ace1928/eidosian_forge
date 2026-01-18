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
def filter_extender_controller_extender_data(json):
    option_list = ['aaa_shared_secret', 'access_point_name', 'admin', 'allowaccess', 'at_dial_script', 'authorized', 'bandwidth_limit', 'billing_start_day', 'cdma_aaa_spi', 'cdma_ha_spi', 'cdma_nai', 'conn_status', 'controller_report', 'description', 'device_id', 'dial_mode', 'dial_status', 'enforce_bandwidth', 'ext_name', 'extension_type', 'ha_shared_secret', 'id', 'ifname', 'initiated_update', 'login_password', 'login_password_change', 'mode', 'modem_passwd', 'modem_type', 'modem1', 'modem2', 'multi_mode', 'name', 'override_allowaccess', 'override_enforce_bandwidth', 'override_login_password_change', 'ppp_auth_protocol', 'ppp_echo_request', 'ppp_password', 'ppp_username', 'primary_ha', 'profile', 'quota_limit_mb', 'redial', 'redundant_intf', 'roaming', 'role', 'secondary_ha', 'sim_pin', 'vdom', 'wan_extension', 'wimax_auth_protocol', 'wimax_carrier', 'wimax_realm']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary