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
def create_sf_connection(module, port=None):
    hostname = module.params['hostname']
    username = module.params['username']
    password = module.params['password']
    if HAS_SF_SDK and hostname and username and password:
        try:
            return_val = ElementFactory.create(hostname, username, password, port=port)
            return return_val
        except Exception:
            raise Exception('Unable to create SF connection')
    else:
        module.fail_json(msg='the python SolidFire SDK module is required')