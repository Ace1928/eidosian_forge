from __future__ import absolute_import, division, print_function
import traceback
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_krbrealm_rest(self, modify):
    api = 'protocols/nfs/kerberos/realms/%s' % self.svm_uuid
    body = {}
    if modify.get('kdc_ip'):
        body['kdc.ip'] = modify['kdc_ip']
    if modify.get('kdc_vendor'):
        body['kdc.vendor'] = modify['kdc_vendor']
    if modify.get('kdc_port'):
        body['kdc.port'] = modify['kdc_port']
    if modify.get('comment'):
        body['comment'] = modify['comment']
    if modify.get('ad_server_ip'):
        body['ad_server.address'] = modify['ad_server_ip']
    if modify.get('ad_server_name'):
        body['ad_server.name'] = modify['ad_server_name']
    if modify.get('admin_server_ip'):
        body['admin_server.address'] = modify['admin_server_ip']
    if modify.get('admin_server_port'):
        body['admin_server.port'] = modify['admin_server_port']
    if modify.get('pw_server_ip'):
        body['password_server.address'] = modify['pw_server_ip']
    if modify.get('pw_server_port'):
        body['password_server.port'] = modify['pw_server_port']
    if modify.get('clock_skew'):
        body['clock_skew'] = modify['clock_skew']
    dummy, error = rest_generic.patch_async(self.rest_api, api, self.parameters['realm'], body)
    if error:
        self.module.fail_json(msg='Error modifying Kerberos Realm %s: %s' % (self.parameters['realm'], to_native(error)))