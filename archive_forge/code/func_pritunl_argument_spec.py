from __future__ import absolute_import, division, print_function
import base64
import hashlib
import hmac
import json
import time
import uuid
from ansible.module_utils.six import iteritems
from ansible.module_utils.urls import open_url
def pritunl_argument_spec():
    return dict(pritunl_url=dict(required=True, type='str'), pritunl_api_token=dict(required=True, type='str', no_log=False), pritunl_api_secret=dict(required=True, type='str', no_log=True), validate_certs=dict(required=False, type='bool', default=True))