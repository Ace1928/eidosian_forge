from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.six.moves.urllib.parse import quote
from ansible.module_utils.six.moves.http_client import HTTPException
import json
def retrieve_credential(module):
    api_base_url = module.params['api_base_url']
    validate_certs = module.params['validate_certs']
    app_id = module.params['app_id']
    query = module.params['query']
    connection_timeout = module.params['connection_timeout']
    query_format = module.params['query_format']
    fail_request_on_password_change = module.params['fail_request_on_password_change']
    client_cert = None
    client_key = None
    path = '/AIMWebService/api/Accounts'
    if 'client_cert' in module.params:
        client_cert = module.params['client_cert']
    if 'client_key' in module.params:
        client_key = module.params['client_key']
    if 'path' in module.params:
        path = module.params['path']
    end_point = '%s?AppId=%s&Query=%s&ConnectionTimeout=%s&QueryFormat=%s&FailRequestOnPasswordChange=%s' % (path, quote(app_id), quote(query), connection_timeout, query_format, fail_request_on_password_change)
    if 'reason' in module.params and module.params['reason'] is not None:
        reason = quote(module.params['reason'])
        end_point = end_point + '&reason=%s' % reason
    result = None
    response = None
    try:
        response = open_url(api_base_url + end_point, method='GET', validate_certs=validate_certs, client_cert=client_cert, client_key=client_key)
    except (HTTPError, HTTPException) as http_exception:
        module.fail_json(msg='Error while retrieving credential.Please validate parameters provided, and permissions for the application and provider in CyberArk.\n*** end_point=%s%s\n ==> %s' % (api_base_url, end_point, to_text(http_exception)), status_code=http_exception.code)
    except Exception as unknown_exception:
        module.fail_json(msg='Unknown error while retrieving credential.\n*** end_point=%s%s\n%s' % (api_base_url, end_point, to_text(unknown_exception)), status_code=-1)
    if response.getcode() == 200:
        try:
            result = json.loads(response.read())
        except Exception as exc:
            module.fail_json(msg='Error obtain cyberark credential result from http body\n%s' % to_text(exc), status_code=-1)
        return (result, response.getcode())
    else:
        module.fail_json(msg='error in end_point=>' + end_point)