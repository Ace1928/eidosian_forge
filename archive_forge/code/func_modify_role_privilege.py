from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_role_privilege(self, privilege, path):
    path = path.replace('/', '%2F')
    api = 'security/roles/%s/%s/privileges' % (self.owner_uuid, self.parameters['name'])
    body = {}
    if privilege.get('access'):
        body['access'] = privilege['access']
    if privilege.get('query'):
        body['query'] = privilege['query']
    dummy, error = rest_generic.patch_async(self.rest_api, api, path, body)
    if error:
        self.module.fail_json(msg='Error modifying privileges for path %s: %s' % (path, to_native(error)), exception=traceback.format_exc())