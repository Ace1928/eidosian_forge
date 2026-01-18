from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_unix_user(self):
    """
        Creates an UNIX user in the specified Vserver

        :return: None
        """
    if self.parameters.get('primary_gid') is None or self.parameters.get('id') is None:
        self.module.fail_json(msg='Error: Missing one or more required parameters for create: (primary_gid, id)')
    user_create = netapp_utils.zapi.NaElement.create_node_with_children('name-mapping-unix-user-create', **{'user-name': self.parameters['name'], 'group-id': str(self.parameters['primary_gid']), 'user-id': str(self.parameters['id'])})
    if self.parameters.get('full_name') is not None:
        user_create.add_new_child('full-name', self.parameters['full_name'])
    try:
        self.server.invoke_successfully(user_create, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error creating UNIX user %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())