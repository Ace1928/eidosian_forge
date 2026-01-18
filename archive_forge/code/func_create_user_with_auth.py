from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def create_user_with_auth(self, application, index):
    """
        creates the user for the given application and authentication_method
        application is now a directory
        :param: application: application to grant access to
        """
    user_create = netapp_utils.zapi.NaElement.create_node_with_children('security-login-create', **{'vserver': self.parameters['vserver'], 'user-name': self.parameters['name'], 'application': application['application'], 'authentication-method': application['authentication_methods'][index], 'role-name': self.parameters.get('role_name')})
    if application.get('second_authentication_method') is not None:
        user_create.add_new_child('second-authentication-method', application['second_authentication_method'])
    if self.parameters.get('set_password') is not None:
        user_create.add_new_child('password', self.parameters.get('set_password'))
    if application['authentication_methods'][0] == 'usm':
        if self.parameters.get('remote_switch_ipaddress') is not None:
            user_create.add_new_child('remote-switch-ipaddress', self.parameters.get('remote_switch_ipaddress'))
        snmpv3_login_info = netapp_utils.zapi.NaElement('snmpv3-login-info')
        if self.parameters.get('authentication_password') is not None:
            snmpv3_login_info.add_new_child('authentication-password', self.parameters['authentication_password'])
        if self.parameters.get('authentication_protocol') is not None:
            snmpv3_login_info.add_new_child('authentication-protocol', self.parameters['authentication_protocol'])
        if self.parameters.get('engine_id') is not None:
            snmpv3_login_info.add_new_child('engine-id', self.parameters['engine_id'])
        if self.parameters.get('privacy_password') is not None:
            snmpv3_login_info.add_new_child('privacy-password', self.parameters['privacy_password'])
        if self.parameters.get('privacy_protocol') is not None:
            snmpv3_login_info.add_new_child('privacy-protocol', self.parameters['privacy_protocol'])
        user_create.add_child_elem(snmpv3_login_info)
    try:
        self.server.invoke_successfully(user_create, enable_tunneling=False)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error creating user %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())