from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def filter_endpoint_control_settings_data(json):
    option_list = ['download_custom_link', 'download_location', 'forticlient_avdb_update_interval', 'forticlient_dereg_unsupported_client', 'forticlient_disconnect_unsupported_client', 'forticlient_ems_rest_api_call_timeout', 'forticlient_keepalive_interval', 'forticlient_offline_grace', 'forticlient_offline_grace_interval', 'forticlient_reg_key', 'forticlient_reg_key_enforce', 'forticlient_reg_timeout', 'forticlient_sys_update_interval', 'forticlient_user_avatar', 'forticlient_warning_interval', 'override']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary