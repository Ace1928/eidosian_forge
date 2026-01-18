from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_kerberos_interface(self):
    """
        Modify kerberos interface.
        """
    api = 'protocols/nfs/kerberos/interfaces'
    body = {'enabled': self.parameters['enabled']}
    if 'keytab_uri' in self.parameters:
        body['keytab_uri'] = self.parameters['keytab_uri']
    if 'organizational_unit' in self.parameters:
        body['organizational_unit'] = self.parameters['organizational_unit']
    if 'service_principal_name' in self.parameters:
        body['spn'] = self.parameters['service_principal_name']
    if 'admin_username' in self.parameters:
        body['user'] = self.parameters['admin_username']
    if 'admin_password' in self.parameters:
        body['password'] = self.parameters['admin_password']
    if 'machine_account' in self.parameters:
        body['machine_account'] = self.parameters['machine_account']
    dummy, error = rest_generic.patch_async(self.rest_api, api, self.uuid, body)
    if error:
        self.module.fail_json(msg='Error modifying kerberos interface %s: %s.' % (self.parameters['interface_name'], to_native(error)), exception=traceback.format_exc())