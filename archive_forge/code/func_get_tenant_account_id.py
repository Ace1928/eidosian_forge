from __future__ import absolute_import, division, print_function
import ansible_collections.netapp.storagegrid.plugins.module_utils.netapp as netapp_utils
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp_module import (
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp import (
def get_tenant_account_id(self):
    api = 'api/v3/grid/accounts'
    params = {'limit': 20}
    params['marker'] = ''
    while params['marker'] is not None:
        list_accounts, error = self.rest_api.get(api, params)
        if error:
            self.module.fail_json(msg=error)
        if len(list_accounts.get('data')) > 0:
            for account in list_accounts['data']:
                if account['name'] == self.parameters['name']:
                    return account['id']
            params['marker'] = list_accounts['data'][-1]['id']
        else:
            params['marker'] = None
    return None