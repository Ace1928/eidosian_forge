from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI, idrac_auth_params
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import strip_substr_dict
def fetch_all_accounts(idrac, accounts_uri):
    all_accounts = idrac.invoke_request('{0}?$expand=*($levels=1)'.format(accounts_uri), 'GET')
    all_accs = all_accounts.json_data.get('Members')
    return all_accs