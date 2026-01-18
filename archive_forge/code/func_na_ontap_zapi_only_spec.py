from __future__ import (absolute_import, division, print_function)
import base64
import logging
import os
import ssl
import time
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils._text import to_native
def na_ontap_zapi_only_spec():
    return dict(hostname=dict(required=True, type='str'), username=dict(required=False, type='str', aliases=['user']), password=dict(required=False, type='str', aliases=['pass'], no_log=True), https=dict(required=False, type='bool', default=False), validate_certs=dict(required=False, type='bool', default=True), http_port=dict(required=False, type='int'), ontapi=dict(required=False, type='int'), use_rest=dict(required=False, type='str', default='never'), feature_flags=dict(required=False, type='dict'), cert_filepath=dict(required=False, type='str'), key_filepath=dict(required=False, type='str', no_log=False))