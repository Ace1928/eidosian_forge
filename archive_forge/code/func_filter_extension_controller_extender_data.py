from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def filter_extension_controller_extender_data(json):
    option_list = ['allowaccess', 'authorized', 'bandwidth_limit', 'description', 'device_id', 'enforce_bandwidth', 'ext_name', 'extension_type', 'firmware_provision_latest', 'id', 'login_password', 'login_password_change', 'name', 'override_allowaccess', 'override_enforce_bandwidth', 'override_login_password_change', 'profile', 'wan_extension']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary