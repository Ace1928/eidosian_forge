from __future__ import absolute_import, division, print_function
from ansible.module_utils.common._collections_compat import Mapping
from ansible_collections.community.crypto.plugins.module_utils.acme.errors import (
def get_account_data(self):
    """
        Retrieve account information. Can only be called when the account
        URI is already known (such as after calling setup_account).
        Return None if the account was deactivated, or a dict otherwise.
        """
    if self.client.account_uri is None:
        raise ModuleFailException('Account URI unknown')
    if self.client.version == 1:
        data = {}
        data['resource'] = 'reg'
        result, info = self.client.send_signed_request(self.client.account_uri, data, fail_on_error=False)
    else:
        data = None
        result, info = self.client.send_signed_request(self.client.account_uri, data, fail_on_error=False)
        if info['status'] >= 400 and result.get('type') == 'urn:ietf:params:acme:error:malformed':
            data = {}
            result, info = self.client.send_signed_request(self.client.account_uri, data, fail_on_error=False)
    if not isinstance(result, Mapping):
        raise ACMEProtocolException(self.client.module, msg='Invalid account data retrieved from ACME server', info=info, content=result)
    if info['status'] in (400, 403) and result.get('type') == 'urn:ietf:params:acme:error:unauthorized':
        return None
    if info['status'] in (400, 404) and result.get('type') == 'urn:ietf:params:acme:error:accountDoesNotExist':
        return None
    if info['status'] < 200 or info['status'] >= 300:
        raise ACMEProtocolException(self.client.module, msg='Error retrieving account data', info=info, content_json=result)
    return result