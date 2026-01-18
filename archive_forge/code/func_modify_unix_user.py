from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_unix_user(self, params):
    user_modify = netapp_utils.zapi.NaElement.create_node_with_children('name-mapping-unix-user-modify', **{'user-name': self.parameters['name']})
    for key in params:
        if key == 'primary_gid':
            user_modify.add_new_child('group-id', str(params['primary_gid']))
        if key == 'id':
            user_modify.add_new_child('user-id', str(params['id']))
        if key == 'full_name':
            user_modify.add_new_child('full-name', params['full_name'])
    try:
        self.server.invoke_successfully(user_modify, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error modifying UNIX user %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())