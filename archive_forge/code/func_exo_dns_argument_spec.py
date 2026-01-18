from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils._text import to_native, to_text
from ansible.module_utils.six import integer_types, string_types
from ansible.module_utils.six.moves import configparser
from ansible.module_utils.urls import fetch_url
def exo_dns_argument_spec():
    return dict(api_key=dict(type='str', default=os.environ.get('CLOUDSTACK_KEY'), no_log=True), api_secret=dict(type='str', default=os.environ.get('CLOUDSTACK_SECRET'), no_log=True), api_timeout=dict(type='int', default=os.environ.get('CLOUDSTACK_TIMEOUT') or 10), api_region=dict(type='str', default=os.environ.get('CLOUDSTACK_REGION') or 'cloudstack'), validate_certs=dict(default=True, type='bool'))