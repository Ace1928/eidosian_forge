from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_text
from ansible.module_utils.six.moves import http_client as httplib
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import quote
import logging
def resolve_group_name_to_id(module):
    group_name = module.params['group_name']
    cyberark_session = module.params['cyberark_session']
    api_base_url = cyberark_session['api_base_url']
    validate_certs = cyberark_session['validate_certs']
    headers = {'Content-Type': 'application/json', 'Authorization': cyberark_session['token'], 'User-Agent': 'CyberArk/1.0 (Ansible; cyberark.pas)'}
    url = construct_url(api_base_url, '/PasswordVault/api/UserGroups?search={pgroupname}'.format(pgroupname=quote(group_name)))
    try:
        response = open_url(url, method='GET', headers=headers, validate_certs=validate_certs, timeout=module.params['timeout'])
        groups = json.loads(response.read())
        group_id = None
        for group in groups['value']:
            if group['groupName'] == group_name:
                if group_id is None:
                    group_id = group['id']
                else:
                    module.fail_json(msg='Found more than one group matching %s. Use vault_id instead' % group_name)
        logging.debug('Resolved group_name %s to ID %s', group_name, group_id)
        return group_id
    except (HTTPError, httplib.HTTPException) as http_exception:
        module.fail_json(msg='Error while looking up group %s.\n*** end_point=%s\n ==> %s' % (group_name, url, to_text(http_exception)), payload={}, headers=headers, status_code=http_exception.code)
    except Exception as unknown_exception:
        module.fail_json(msg='Unknown error while looking up group %s.\n*** end_point=%s\n%s' % (group_name, url, to_text(unknown_exception)), payload={}, headers=headers, status_code=-1)