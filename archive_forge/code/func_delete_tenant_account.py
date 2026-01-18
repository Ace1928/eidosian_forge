from __future__ import absolute_import, division, print_function
import ansible_collections.netapp.storagegrid.plugins.module_utils.netapp as netapp_utils
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp_module import (
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp import (
def delete_tenant_account(self, account_id):
    api = 'api/v3/grid/accounts/' + account_id
    self.data = None
    response, error = self.rest_api.delete(api, self.data)
    if error:
        self.module.fail_json(msg=error)