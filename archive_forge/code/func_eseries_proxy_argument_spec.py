from __future__ import absolute_import, division, print_function
import json
import random
import mimetypes
from pprint import pformat
from ansible.module_utils import six
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.six.moves.urllib.error import HTTPError, URLError
from ansible.module_utils.urls import open_url
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils._text import to_native
def eseries_proxy_argument_spec():
    """Retrieve a base argument specification common to all NetApp E-Series modules for proxy specific tasks"""
    argument_spec = basic_auth_argument_spec()
    argument_spec.update(dict(api_username=dict(type='str', required=True), api_password=dict(type='str', required=True, no_log=True), api_url=dict(type='str', required=True), validate_certs=dict(type='bool', required=False, default=True)))
    return argument_spec