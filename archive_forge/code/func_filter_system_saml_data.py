from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def filter_system_saml_data(json):
    option_list = ['artifact_resolution_url', 'binding_protocol', 'cert', 'default_login_page', 'default_profile', 'entity_id', 'idp_artifact_resolution_url', 'idp_cert', 'idp_entity_id', 'idp_single_logout_url', 'idp_single_sign_on_url', 'life', 'portal_url', 'role', 'server_address', 'service_providers', 'single_logout_url', 'single_sign_on_url', 'status', 'tolerance']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary