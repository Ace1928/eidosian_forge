from __future__ import absolute_import, division, print_function
import base64
import hashlib
import hmac
import json
import time
import uuid
from ansible.module_utils.six import iteritems
from ansible.module_utils.urls import open_url
def get_pritunl_settings(module):
    """
    Helper function to set required Pritunl request params from module arguments.
    """
    return {'api_token': module.params.get('pritunl_api_token'), 'api_secret': module.params.get('pritunl_api_secret'), 'base_url': module.params.get('pritunl_url'), 'validate_certs': module.params.get('validate_certs')}