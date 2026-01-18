from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def rename_subnet(self):
    """
        TODO
        """
    options = {'subnet-name': self.parameters.get('from_name'), 'new-name': self.parameters.get('name')}
    subnet_rename = netapp_utils.zapi.NaElement.create_node_with_children('net-subnet-rename', **options)
    if self.parameters.get('ipspace'):
        subnet_rename.add_new_child('ipspace', self.parameters.get('ipspace'))
    try:
        self.server.invoke_successfully(subnet_rename, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error renaming subnet %s: %s' % (self.parameters.get('name'), to_native(error)), exception=traceback.format_exc())