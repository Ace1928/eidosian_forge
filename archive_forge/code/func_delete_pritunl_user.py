from __future__ import absolute_import, division, print_function
import base64
import hashlib
import hmac
import json
import time
import uuid
from ansible.module_utils.six import iteritems
from ansible.module_utils.urls import open_url
def delete_pritunl_user(api_token, api_secret, base_url, organization_id, user_id, validate_certs=True):
    response = _delete_pritunl_user(api_token=api_token, api_secret=api_secret, base_url=base_url, organization_id=organization_id, user_id=user_id, validate_certs=validate_certs)
    if response.getcode() != 200:
        raise PritunlException('Could not remove user %s from organization %s from Pritunl' % (user_id, organization_id))
    return json.loads(response.read())