from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI, idrac_auth_params
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import strip_substr_dict
def get_all_accounts(idrac, account_uri):
    all_accs = fetch_all_accounts(idrac, account_uri)
    idrac_list = []
    for acc in all_accs:
        if acc.get('UserName') != '':
            acc.pop('Links', None)
            acc_dets_json_data = strip_substr_dict(acc)
            if acc_dets_json_data.get('Oem') is not None:
                acc_dets_json_data['Oem']['Dell'] = strip_substr_dict(acc_dets_json_data['Oem']['Dell'])
            idrac_list.append(acc_dets_json_data)
    return idrac_list