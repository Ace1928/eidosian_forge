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
def filter_system_sdn_connector_data(json):
    option_list = ['access_key', 'alt_resource_ip', 'api_key', 'azure_region', 'client_id', 'client_secret', 'compartment_id', 'compartment_list', 'compute_generation', 'domain', 'external_account_list', 'external_ip', 'forwarding_rule', 'gcp_project', 'gcp_project_list', 'group_name', 'ha_status', 'ibm_region', 'ibm_region_gen1', 'ibm_region_gen2', 'key_passwd', 'login_endpoint', 'name', 'nic', 'oci_cert', 'oci_fingerprint', 'oci_region', 'oci_region_list', 'oci_region_type', 'password', 'private_key', 'proxy', 'region', 'resource_group', 'resource_url', 'route', 'route_table', 'secret_key', 'secret_token', 'server', 'server_ca_cert', 'server_cert', 'server_list', 'server_port', 'service_account', 'status', 'subscription_id', 'tenant_id', 'type', 'update_interval', 'use_metadata_iam', 'user_id', 'username', 'vcenter_password', 'vcenter_server', 'vcenter_username', 'verify_certificate', 'vpc_id']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary