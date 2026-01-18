from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def create_zapi_api(self, api):
    """
        Create an the ZAPI API request for fpolicy modify and create
        :return: ZAPI API object
        """
    fpolicy_ext_engine_obj = netapp_utils.zapi.NaElement(api)
    fpolicy_ext_engine_obj.add_new_child('engine-name', self.parameters['name'])
    fpolicy_ext_engine_obj.add_new_child('port-number', self.na_helper.get_value_for_int(from_zapi=False, value=self.parameters['port']))
    fpolicy_ext_engine_obj.add_new_child('ssl-option', self.parameters['ssl_option'])
    primary_servers_obj = netapp_utils.zapi.NaElement('primary-servers')
    for primary_server in self.parameters['primary_servers']:
        primary_servers_obj.add_new_child('ip-address', primary_server)
    fpolicy_ext_engine_obj.add_child_elem(primary_servers_obj)
    if 'secondary_servers' in self.parameters:
        secondary_servers_obj = netapp_utils.zapi.NaElement('secondary-servers')
        for secondary_server in self.parameters['secondary_servers']:
            primary_servers_obj.add_new_child('ip-address', secondary_server)
        fpolicy_ext_engine_obj.add_child_elem(secondary_servers_obj)
    if 'is_resiliency_enabled' in self.parameters:
        fpolicy_ext_engine_obj.add_new_child('is-resiliency-enabled', self.na_helper.get_value_for_bool(from_zapi=False, value=self.parameters['is_resiliency_enabled']))
    if 'resiliency_directory_path' in self.parameters:
        fpolicy_ext_engine_obj.add_new_child('resiliency-directory-path', self.parameters['resiliency_directory_path'])
    if 'max_connection_retries' in self.parameters:
        fpolicy_ext_engine_obj.add_new_child('max-connection-retries', self.na_helper.get_value_for_int(from_zapi=False, value=self.parameters['max_connection_retries']))
    if 'max_server_reqs' in self.parameters:
        fpolicy_ext_engine_obj.add_new_child('max-server-requests', self.na_helper.get_value_for_int(from_zapi=False, value=self.parameters['max_server_reqs']))
    if 'recv_buffer_size' in self.parameters:
        fpolicy_ext_engine_obj.add_new_child('recv-buffer-size', self.na_helper.get_value_for_int(from_zapi=False, value=self.parameters['recv_buffer_size']))
    if 'send_buffer_size' in self.parameters:
        fpolicy_ext_engine_obj.add_new_child('send-buffer-size', self.na_helper.get_value_for_int(from_zapi=False, value=self.parameters['send_buffer_size']))
    if 'certificate_ca' in self.parameters:
        fpolicy_ext_engine_obj.add_new_child('certificate-ca', self.parameters['certificate_ca'])
    if 'certificate_common_name' in self.parameters:
        fpolicy_ext_engine_obj.add_new_child('certificate-common-name', self.parameters['certificate_common_name'])
    if 'certificate_serial' in self.parameters:
        fpolicy_ext_engine_obj.add_new_child('certificate-serial', self.parameters['certificate_serial'])
    if 'extern_engine_type' in self.parameters:
        fpolicy_ext_engine_obj.add_new_child('extern-engine-type', self.parameters['extern_engine_type'])
    return fpolicy_ext_engine_obj