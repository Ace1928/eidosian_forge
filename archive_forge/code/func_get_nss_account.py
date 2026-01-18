from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp_module import NetAppModule
def get_nss_account(self):
    response, err, dummy = self.rest_api.send_request('GET', '%s/accounts' % self.rest_api.api_root_path, None, header=self.headers)
    if err is not None:
        self.module.fail_json(changed=False, msg='Error: unexpected response on getting nss account: %s, %s' % (str(err), str(response)))
    if response is None:
        return None
    nss_accounts = []
    if response.get('nssAccounts'):
        nss_accounts = response['nssAccounts']
    if len(nss_accounts) == 0:
        return None
    result = dict()
    for account in nss_accounts:
        if account['nssUserName'] == self.parameters['username']:
            if self.parameters.get('public_id') and self.parameters['public_id'] != account['publicId']:
                self.module.fail_json(changed=False, msg="Error: public_id '%s' does not match username." % account['publicId'])
            else:
                self.parameters['public_id'] = account['publicId']
            result['name'] = account['accountName']
            result['user_name'] = account['nssUserName']
            result['vsa_list'] = account['vsaList']
            return result
    return None