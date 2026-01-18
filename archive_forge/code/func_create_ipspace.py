from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_ipspace(self):
    """
        Create ipspace
        :return: None
        """
    if self.use_rest:
        api = 'network/ipspaces'
        body = {'name': self.parameters['name']}
        dummy, error = rest_generic.post_async(self.rest_api, api, body)
        if error:
            self.module.fail_json(msg='Error provisioning ipspace %s: %s' % (self.parameters['name'], error))
    else:
        ipspace_create = netapp_utils.zapi.NaElement.create_node_with_children('net-ipspaces-create', **{'ipspace': self.parameters['name']})
        try:
            self.server.invoke_successfully(ipspace_create, enable_tunneling=False)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error provisioning ipspace %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())