from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def lock_given_user(self):
    """
        locks the user
        """
    user_lock = netapp_utils.zapi.NaElement.create_node_with_children('security-login-lock', **{'vserver': self.parameters['vserver'], 'user-name': self.parameters['name']})
    try:
        self.server.invoke_successfully(user_lock, enable_tunneling=False)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error locking user %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())