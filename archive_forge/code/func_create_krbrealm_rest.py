from __future__ import absolute_import, division, print_function
import traceback
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_krbrealm_rest(self):
    api = 'protocols/nfs/kerberos/realms'
    body = {'name': self.parameters['realm'], 'svm.name': self.parameters['vserver'], 'kdc.ip': self.parameters['kdc_ip'], 'kdc.vendor': self.parameters['kdc_vendor']}
    if self.parameters.get('kdc_port'):
        body['kdc.port'] = self.parameters['kdc_port']
    if self.parameters.get('comment'):
        body['comment'] = self.parameters['comment']
    if self.parameters.get('ad_server_ip'):
        body['ad_server.address'] = self.parameters['ad_server_ip']
    if self.parameters.get('ad_server_name'):
        body['ad_server.name'] = self.parameters['ad_server_name']
    if self.parameters.get('admin_server_port'):
        body['admin_server.port'] = self.parameters['admin_server_port']
    if self.parameters.get('pw_server_port'):
        body['password_server.port'] = self.parameters['pw_server_port']
    if self.parameters.get('clock_skew'):
        body['clock_skew'] = self.parameters['clock_skew']
    if self.parameters.get('admin_server_ip'):
        body['admin_server.address'] = self.parameters['admin_server_ip']
    if self.parameters.get('pw_server_ip'):
        body['password_server.address'] = self.parameters['pw_server_ip']
    dummy, error = rest_generic.post_async(self.rest_api, api, body)
    if error:
        self.module.fail_json(msg='Error creating Kerberos Realm configuration %s: %s' % (self.parameters['realm'], to_native(error)))