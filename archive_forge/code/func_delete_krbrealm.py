from __future__ import absolute_import, division, print_function
import traceback
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def delete_krbrealm(self):
    """
        Delete Kerberos Realm configuration
        """
    if self.use_rest:
        return self.delete_krbrealm_rest()
    krbrealm_delete = netapp_utils.zapi.NaElement.create_node_with_children('kerberos-realm-delete', **{'realm': self.parameters['realm']})
    try:
        self.server.invoke_successfully(krbrealm_delete, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as errcatch:
        self.module.fail_json(msg='Error deleting Kerberos Realm configuration %s: %s' % (self.parameters['realm'], to_native(errcatch)), exception=traceback.format_exc())