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
def filter_system_automation_action_data(json):
    option_list = ['accprofile', 'action_type', 'alicloud_access_key_id', 'alicloud_access_key_secret', 'alicloud_account_id', 'alicloud_function', 'alicloud_function_authorization', 'alicloud_function_domain', 'alicloud_region', 'alicloud_service', 'alicloud_version', 'aws_api_id', 'aws_api_key', 'aws_api_path', 'aws_api_stage', 'aws_domain', 'aws_region', 'azure_api_key', 'azure_app', 'azure_domain', 'azure_function', 'azure_function_authorization', 'delay', 'description', 'email_body', 'email_from', 'email_subject', 'email_to', 'execute_security_fabric', 'forticare_email', 'fos_message', 'gcp_function', 'gcp_function_domain', 'gcp_function_region', 'gcp_project', 'headers', 'http_body', 'http_headers', 'message_type', 'method', 'minimum_interval', 'name', 'output_size', 'port', 'protocol', 'replacement_message', 'replacemsg_group', 'required', 'script', 'sdn_connector', 'security_tag', 'system_action', 'timeout', 'tls_certificate', 'uri', 'verify_host_cert']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary