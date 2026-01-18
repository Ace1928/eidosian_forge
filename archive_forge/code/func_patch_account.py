from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def patch_account(self, owner_uuid, username, body):
    query = {'name': self.parameters['name'], 'owner.uuid': owner_uuid}
    api = 'security/accounts/%s/%s' % (owner_uuid, username)
    dummy, result = self.rest_api.patch(api, body, query)
    return result