from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible.module_utils.urls import Request, SSLValidationError, ConnectionError
from ansible.module_utils.parsing.convert_bool import boolean as strtobool
from ansible.module_utils.six import PY2
from ansible.module_utils.six import raise_from, string_types
from ansible.module_utils.six.moves import StringIO
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.six.moves.http_cookiejar import CookieJar
from ansible.module_utils.six.moves.urllib.parse import urlparse, urlencode, quote
from ansible.module_utils.six.moves.configparser import ConfigParser, NoOptionError
from socket import getaddrinfo, IPPROTO_TCP
import time
import re
from json import loads, dumps
from os.path import isfile, expanduser, split, join, exists, isdir
from os import access, R_OK, getcwd, environ
def copy_item(self, existing_item, copy_from_name_or_id, new_item_name, endpoint=None, item_type='unknown', copy_lookup_data=None):
    if existing_item is not None:
        self.warn('A {0} with the name {1} already exists.'.format(item_type, new_item_name))
        self.json_output['changed'] = False
        self.json_output['copied'] = False
        return existing_item
    copy_from_lookup = self.get_one(endpoint, name_or_id=copy_from_name_or_id, **{'data': copy_lookup_data})
    if copy_from_lookup is None:
        self.fail_json(msg='A {0} with the name {1} was not able to be found.'.format(item_type, copy_from_name_or_id))
    if item_type == 'workflow_job_template':
        copy_get_check = self.get_endpoint(copy_from_lookup['related']['copy'])
        if copy_get_check['status_code'] in [200]:
            if copy_get_check['json']['can_copy'] and copy_get_check['json']['can_copy_without_user_input'] and (not copy_get_check['json']['templates_unable_to_copy']) and (not copy_get_check['json']['credentials_unable_to_copy']) and (not copy_get_check['json']['inventories_unable_to_copy']):
                self.json_output['copy_checks'] = 'passed'
            else:
                self.fail_json(msg='Unable to copy {0} {1} error: {2}'.format(item_type, copy_from_name_or_id, copy_get_check))
        else:
            self.fail_json(msg='Error accessing {0} {1} error: {2} '.format(item_type, copy_from_name_or_id, copy_get_check))
    response = self.post_endpoint(copy_from_lookup['related']['copy'], **{'data': {'name': new_item_name}})
    if response['status_code'] in [201]:
        self.json_output['id'] = response['json']['id']
        self.json_output['changed'] = True
        self.json_output['copied'] = True
        new_existing_item = response['json']
    elif 'json' in response and '__all__' in response['json']:
        self.fail_json(msg='Unable to create {0} {1}: {2}'.format(item_type, new_item_name, response['json']['__all__'][0]))
    elif 'json' in response:
        self.fail_json(msg='Unable to create {0} {1}: {2}'.format(item_type, new_item_name, response['json']))
    else:
        self.fail_json(msg='Unable to create {0} {1}: {2}'.format(item_type, new_item_name, response['status_code']))
    return new_existing_item