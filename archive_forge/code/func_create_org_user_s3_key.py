from __future__ import absolute_import, division, print_function
import ansible_collections.netapp.storagegrid.plugins.module_utils.netapp as netapp_utils
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp import SGRestAPI
def create_org_user_s3_key(self, user_id):
    api = 'api/v3/org/users/current-user/s3-access-keys'
    if user_id:
        api = 'api/v3/org/users/%s/s3-access-keys' % user_id
    response, error = self.rest_api.post(api, self.data)
    if error:
        self.module.fail_json(msg=error)
    return response['data']