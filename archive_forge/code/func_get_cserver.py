from __future__ import absolute_import, division, print_function
import json
import os
import random
import mimetypes
from pprint import pformat
from ansible.module_utils import six
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.six.moves.urllib.error import HTTPError, URLError
from ansible.module_utils.urls import open_url
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils._text import to_native
import ssl
def get_cserver(connection, is_rest=False):
    if not is_rest:
        return get_cserver_zapi(connection)
    params = {'fields': 'type'}
    api = 'private/cli/vserver'
    json, error = connection.get(api, params)
    if json is None or error is not None:
        return None
    vservers = json.get('records')
    if vservers is not None:
        for vserver in vservers:
            if vserver['type'] == 'admin':
                return vserver['vserver']
        if len(vservers) == 1:
            return vservers[0]['vserver']
    return None