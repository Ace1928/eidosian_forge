from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_or_modify_qtree_zapi(self, zapi_request_name, error_message):
    options = {'qtree': self.parameters['name'], 'volume': self.parameters['flexvol_name']}
    if self.parameters.get('export_policy'):
        options['export-policy'] = self.parameters['export_policy']
    if self.parameters.get('security_style'):
        options['security-style'] = self.parameters['security_style']
    if self.parameters.get('oplocks'):
        options['oplocks'] = self.parameters['oplocks']
    if self.parameters.get('unix_permissions'):
        options['mode'] = self.parameters['unix_permissions']
    zapi_request = netapp_utils.zapi.NaElement.create_node_with_children(zapi_request_name, **options)
    try:
        self.server.invoke_successfully(zapi_request, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg=error_message % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())