from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def delete_ipspace(self):
    """
        Destroy ipspace
        :return: None
        """
    if self.use_rest:
        api = 'network/ipspaces'
        dummy, error = rest_generic.delete_async(self.rest_api, api, self.uuid)
        if error:
            self.module.fail_json(msg='Error removing ipspace %s: %s' % (self.parameters['name'], error))
    else:
        ipspace_destroy = netapp_utils.zapi.NaElement.create_node_with_children('net-ipspaces-destroy', **{'ipspace': self.parameters['name']})
        try:
            self.server.invoke_successfully(ipspace_destroy, enable_tunneling=False)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error removing ipspace %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())