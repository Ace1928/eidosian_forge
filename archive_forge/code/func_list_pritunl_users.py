from __future__ import absolute_import, division, print_function
import base64
import hashlib
import hmac
import json
import time
import uuid
from ansible.module_utils.six import iteritems
from ansible.module_utils.urls import open_url
def list_pritunl_users(api_token, api_secret, base_url, organization_id, validate_certs=True, filters=None):
    users = []
    response = _get_pritunl_users(api_token=api_token, api_secret=api_secret, base_url=base_url, validate_certs=validate_certs, organization_id=organization_id)
    if response.getcode() != 200:
        raise PritunlException('Could not retrieve users from Pritunl')
    else:
        for user in json.loads(response.read()):
            if filters is None:
                users.append(user)
            elif not any((filter_val != user[filter_key] for filter_key, filter_val in iteritems(filters))):
                users.append(user)
    return users