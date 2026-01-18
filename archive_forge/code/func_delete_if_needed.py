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
def delete_if_needed(self, existing_item, on_delete=None, auto_exit=True):
    if existing_item:
        try:
            item_url = existing_item['url']
            item_type = existing_item['type']
            item_id = existing_item['id']
            item_name = self.get_item_name(existing_item, allow_unknown=True)
        except KeyError as ke:
            self.fail_json(msg='Unable to process delete of item due to missing data {0}'.format(ke))
        response = self.delete_endpoint(item_url)
        if response['status_code'] in [202, 204]:
            if on_delete:
                on_delete(self, response['json'])
            self.json_output['changed'] = True
            self.json_output['id'] = item_id
            self.exit_json(**self.json_output)
            if auto_exit:
                self.exit_json(**self.json_output)
            else:
                return self.json_output
        elif 'json' in response and '__all__' in response['json']:
            self.fail_json(msg='Unable to delete {0} {1}: {2}'.format(item_type, item_name, response['json']['__all__'][0]))
        elif 'json' in response:
            if 'error' in response['json']:
                self.fail_json(msg='Unable to delete {0} {1}: {2}'.format(item_type, item_name, response['json']['error']))
            else:
                self.fail_json(msg='Unable to delete {0} {1}: {2}'.format(item_type, item_name, response['json']))
        else:
            self.fail_json(msg='Unable to delete {0} {1}: {2}'.format(item_type, item_name, response['status_code']))
    elif auto_exit:
        self.exit_json(**self.json_output)
    else:
        return self.json_output