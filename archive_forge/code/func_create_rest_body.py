from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def create_rest_body(self):
    """
        Create an fPolicy body for a create operation
        :return: body as dict
        """
    body = {'vserver': self.parameters['vserver'], 'engine-name': self.parameters['name'], 'primary_servers': self.parameters['primary_servers'], 'port': self.parameters['port'], 'ssl_option': self.parameters['ssl_option']}
    list_of_options = ['secondary_servers', 'is_resiliency_enabled', 'resiliency_directory_path', 'max_connection_retries', 'max_server_reqs', 'recv_buffer_size', 'send_buffer_size', 'certificate_ca', 'certificate_common_name', 'certificate_serial', 'extern_engine_type']
    for option in list_of_options:
        if option in self.parameters:
            body[option] = self.parameters[option]
    return body