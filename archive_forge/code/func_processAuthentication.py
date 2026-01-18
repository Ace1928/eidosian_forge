from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.six.moves.http_client import HTTPException
import json
def processAuthentication(module):
    api_base_url = module.params['api_base_url']
    validate_certs = module.params['validate_certs']
    username = module.params['username']
    password = module.params['password']
    new_password = module.params['new_password']
    use_radius = module.params['use_radius_authentication']
    use_ldap = module.params['use_ldap_authentication']
    use_windows = module.params['use_windows_authentication']
    use_cyberark = module.params['use_cyberark_authentication']
    state = module.params['state']
    cyberark_session = module.params['cyberark_session']
    concurrentSession = module.params['concurrentSession']
    timeout = module.params['timeout']
    if module.check_mode and new_password is not None:
        new_password = None
    headers = {'Content-Type': 'application/json', 'User-Agent': 'CyberArk/1.0 (Ansible; cyberark.pas)'}
    payload = ''
    if state == 'present':
        if use_ldap:
            end_point = '/PasswordVault/API/Auth/LDAP/Logon'
            payload_dict = {'username': username, 'password': password}
        elif use_radius:
            end_point = '/PasswordVault/API/Auth/radius/Logon'
        elif use_windows:
            end_point = '/PasswordVault/API/auth/Windows/Logon'
        else:
            use_cyberark = True
            end_point = '/PasswordVault/API/Auth/CyberArk/Logon'
        payload_dict = {'username': username, 'password': password}
        if new_password is not None and use_cyberark:
            payload_dict['newPassword'] = new_password
        if concurrentSession:
            payload_dict['concurrentSession'] = True
        payload = json.dumps(payload_dict)
    else:
        api_base_url = cyberark_session['api_base_url']
        validate_certs = cyberark_session['validate_certs']
        headers['Authorization'] = cyberark_session['token']
        end_point = '/PasswordVault/API/Auth/Logoff'
    result = None
    changed = False
    response = None
    try:
        response = open_url(api_base_url + end_point, method='POST', headers=headers, data=payload, validate_certs=validate_certs, timeout=timeout)
    except (HTTPError, HTTPException) as http_exception:
        module.fail_json(msg='Error while performing authentication.Please validate parameters provided, and ability to logon to CyberArk.\n*** end_point=%s%s\n ==> %s' % (api_base_url, end_point, to_text(http_exception)), payload=payload, headers=headers, status_code=http_exception.code)
    except Exception as unknown_exception:
        module.fail_json(msg='Unknown error while performing authentication.\n*** end_point=%s%s\n%s' % (api_base_url, end_point, to_text(unknown_exception)), payload=payload, headers=headers, status_code=-1)
    if response.getcode() == 200:
        if state == 'present':
            token = ''
            try:
                token = str(json.loads(response.read()))
            except Exception as e:
                module.fail_json(msg='Error obtaining token\n%s' % to_text(e), payload=payload, headers=headers, status_code=-1)
            result = {'cyberark_session': {'token': token, 'api_base_url': api_base_url, 'validate_certs': validate_certs}}
            if new_password is not None:
                changed = True
        else:
            result = {'cyberark_session': {}}
        return (changed, result, response.getcode())
    else:
        module.fail_json(msg='error in end_point=>' + end_point, headers=headers)