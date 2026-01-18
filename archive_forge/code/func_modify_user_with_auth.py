from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def modify_user_with_auth(self, application, index):
    """
        Modify user
        application is now a dict
        """
    user_modify = netapp_utils.zapi.NaElement.create_node_with_children('security-login-modify', **{'vserver': self.parameters['vserver'], 'user-name': self.parameters['name'], 'application': application['application'], 'authentication-method': application['authentication_methods'][index], 'role-name': self.parameters.get('role_name')})
    try:
        self.server.invoke_successfully(user_modify, enable_tunneling=False)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error modifying user %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())