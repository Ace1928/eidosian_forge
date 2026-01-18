from __future__ import absolute_import, division, print_function
import base64
import hashlib
import hmac
import json
import time
import uuid
from ansible.module_utils.six import iteritems
from ansible.module_utils.urls import open_url
def post_pritunl_organization(api_token, api_secret, base_url, organization_name, validate_certs=True):
    response = _post_pritunl_organization(api_token=api_token, api_secret=api_secret, base_url=base_url, organization_data={'name': organization_name}, validate_certs=validate_certs)
    if response.getcode() != 200:
        raise PritunlException('Could not add organization %s to Pritunl' % organization_name)
    return json.loads(response.read())