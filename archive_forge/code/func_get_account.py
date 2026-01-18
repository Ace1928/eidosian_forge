from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.six.moves.urllib.parse import quote
from ansible.module_utils.six.moves.http_client import HTTPException
import json
import logging
def get_account(module):
    logging.debug('Finding Account')
    identified_by_fields = module.params['identified_by'].split(',')
    logging.debug('Identified_by: %s', json.dumps(identified_by_fields))
    safe_filter = quote('safeName eq ') + quote(module.params['safe']) if 'safe' in module.params and module.params['safe'] is not None else None
    search_string = None
    for field in identified_by_fields:
        if field not in ansible_specific_parameters:
            search_string = '%s%s' % (search_string + ' ' if search_string is not None else '', deep_get(module.params, field, 'NOT FOUND', False))
    logging.debug('Search_String => %s', search_string)
    logging.debug('Safe Filter => %s', safe_filter)
    cyberark_session = module.params['cyberark_session']
    api_base_url = cyberark_session['api_base_url']
    validate_certs = cyberark_session['validate_certs']
    end_point = None
    if search_string is not None and safe_filter is not None:
        end_point = '/PasswordVault/api/accounts?filter=%s&search=%s' % (safe_filter, quote(search_string.lstrip()))
    elif search_string is not None:
        end_point = '/PasswordVault/api/accounts?search=%s' % search_string.lstrip()
    else:
        end_point = '/PasswordVault/api/accounts?filter=%s' % safe_filter
    logging.debug('End Point => %s', end_point)
    headers = {'Content-Type': 'application/json', 'Authorization': cyberark_session['token'], 'User-Agent': 'CyberArk/1.0 (Ansible; cyberark.pas)'}
    try:
        logging.debug('Executing: ' + api_base_url + end_point)
        response = open_url(api_base_url + end_point, method='GET', headers=headers, validate_certs=validate_certs)
        result_string = response.read()
        accounts_data = json.loads(result_string)
        logging.debug('RESULT => %s', json.dumps(accounts_data))
        if accounts_data['count'] == 0:
            return (False, None, response.getcode())
        else:
            how_many = 0
            first_record_found = None
            for account_record in accounts_data['value']:
                logging.debug('Acct Record => %s', json.dumps(account_record))
                found = False
                for field in identified_by_fields:
                    record_field_value = deep_get(account_record, field, 'NOT FOUND')
                    logging.debug('Comparing field %s | record_field_name=%s  record_field_value=%s   module.params_value=%s', field, field, record_field_value, deep_get(module.params, field, 'NOT FOUND'))
                    if record_field_value != 'NOT FOUND' and record_field_value == deep_get(module.params, field, 'NOT FOUND', False):
                        found = True
                    else:
                        found = False
                        break
                if found:
                    how_many = how_many + 1
                    if first_record_found is None:
                        first_record_found = account_record
            logging.debug('How Many: %d  First Record Found => %s', how_many, json.dumps(first_record_found))
            if how_many > 1:
                module.fail_json(msg='Error while performing get_account. Too many rows (%d) found matching your criteria!' % how_many)
            else:
                return (how_many == 1, first_record_found, response.getcode())
    except (HTTPError, HTTPException) as http_exception:
        if http_exception.code == 404:
            return (False, None, http_exception.code)
        else:
            module.fail_json(msg='Error while performing get_account.Please validate parameters provided.\n*** end_point=%s%s\n ==> %s' % (api_base_url, end_point, to_text(http_exception)), headers=headers, status_code=http_exception.code)
    except Exception as unknown_exception:
        module.fail_json(msg='Unknown error while performing get_account.\n*** end_point=%s%s\n%s' % (api_base_url, end_point, to_text(unknown_exception)), headers=headers, status_code=-1)