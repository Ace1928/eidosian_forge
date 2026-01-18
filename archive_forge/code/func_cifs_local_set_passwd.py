from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver
def cifs_local_set_passwd(self):
    """
        :return: None
        """
    if self.use_rest:
        return self.cifs_local_set_passwd_rest()
    cifs_local_set_passwd = netapp_utils.zapi.NaElement('cifs-local-user-set-password')
    cifs_local_set_passwd.add_new_child('user-name', self.parameters['user_name'])
    cifs_local_set_passwd.add_new_child('user-password', self.parameters['user_password'])
    try:
        self.server.invoke_successfully(cifs_local_set_passwd, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as e:
        self.module.fail_json(msg='Error setting password for local CIFS user %s on vserver %s: %s' % (self.parameters['user_name'], self.parameters['vserver'], to_native(e)), exception=traceback.format_exc())