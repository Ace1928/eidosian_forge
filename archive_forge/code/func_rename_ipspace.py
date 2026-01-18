from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def rename_ipspace(self):
    """
        Rename an ipspace
        :return: Nothing
        """
    if self.use_rest:
        api = 'network/ipspaces'
        body = {'name': self.parameters['name']}
        dummy, error = rest_generic.patch_async(self.rest_api, api, self.uuid, body)
        if error:
            self.module.fail_json(msg='Error renaming ipspace %s: %s' % (self.parameters['from_name'], error))
    else:
        ipspace_rename = netapp_utils.zapi.NaElement.create_node_with_children('net-ipspaces-rename', **{'ipspace': self.parameters['from_name'], 'new-name': self.parameters['name']})
        try:
            self.server.invoke_successfully(ipspace_rename, enable_tunneling=False)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error renaming ipspace %s: %s' % (self.parameters['from_name'], to_native(error)), exception=traceback.format_exc())