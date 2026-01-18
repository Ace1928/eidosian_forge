from __future__ import absolute_import, division, print_function
import base64
import hashlib
import hmac
import json
import time
import uuid
from ansible.module_utils.six import iteritems
from ansible.module_utils.urls import open_url
def pritunl_auth_request(api_token, api_secret, base_url, method, path, validate_certs=True, headers=None, data=None):
    """
    Send an API call to a Pritunl server.
    Taken from https://pritunl.com/api and adapted to work with Ansible open_url
    """
    auth_timestamp = str(int(time.time()))
    auth_nonce = uuid.uuid4().hex
    auth_string = '&'.join([api_token, auth_timestamp, auth_nonce, method.upper(), path])
    auth_signature = base64.b64encode(hmac.new(api_secret.encode('utf-8'), auth_string.encode('utf-8'), hashlib.sha256).digest())
    auth_headers = {'Auth-Token': api_token, 'Auth-Timestamp': auth_timestamp, 'Auth-Nonce': auth_nonce, 'Auth-Signature': auth_signature}
    if headers:
        auth_headers.update(headers)
    try:
        uri = '%s%s' % (base_url, path)
        return open_url(uri, method=method.upper(), headers=auth_headers, data=data, validate_certs=validate_certs)
    except Exception as e:
        raise PritunlException(e)