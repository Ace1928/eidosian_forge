from __future__ import absolute_import, division, print_function
import ansible_collections.netapp.storagegrid.plugins.module_utils.netapp as netapp_utils
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp_module import (
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp import (
def set_tenant_root_password(self, account_id):
    api = 'api/v3/grid/accounts/%s/change-password' % account_id
    response, error = self.rest_api.post(api, self.pw_change)
    if error:
        self.module.fail_json(msg=error['text'])