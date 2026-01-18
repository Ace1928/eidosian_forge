from __future__ import absolute_import, division, print_function
import base64
import hashlib
import hmac
import json
import time
import uuid
from ansible.module_utils.six import iteritems
from ansible.module_utils.urls import open_url
def list_pritunl_organizations(api_token, api_secret, base_url, validate_certs=True, filters=None):
    orgs = []
    response = _get_pritunl_organizations(api_token=api_token, api_secret=api_secret, base_url=base_url, validate_certs=validate_certs)
    if response.getcode() != 200:
        raise PritunlException('Could not retrieve organizations from Pritunl')
    else:
        for org in json.loads(response.read()):
            if filters is None:
                orgs.append(org)
            elif not any((filter_val != org[filter_key] for filter_key, filter_val in iteritems(filters))):
                orgs.append(org)
    return orgs