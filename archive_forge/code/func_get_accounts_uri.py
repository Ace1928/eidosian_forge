from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI, idrac_auth_params
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import strip_substr_dict
def get_accounts_uri(idrac):
    try:
        account_path = idrac.invoke_request(ACCOUNT, 'GET')
        account_service = account_path.json_data.get('AccountService').get('@odata.id')
        accounts = idrac.invoke_request(account_service, 'GET')
        accounts_uri = accounts.json_data.get('Accounts').get('@odata.id')
    except HTTPError:
        accounts_uri = '/redfish/v1/AccountService/Accounts'
    return accounts_uri