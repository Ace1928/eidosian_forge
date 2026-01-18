from __future__ import (absolute_import, division, print_function)
from base64 import b64encode
from email.utils import formatdate
import re
import json
import hashlib
from ansible.module_utils.six import iteritems
from ansible.module_utils.six.moves.urllib.parse import urlparse, urlencode
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.basic import env_fallback
def intersight_call(self, http_method='', resource_path='', query_params=None, body=None, moid=None, name=None):
    """
        Invoke the Intersight API

        :param resource_path: intersight resource path e.g. '/ntp/Policies'
        :param query_params: dictionary object with query string parameters as key/value pairs
        :param body: dictionary object with intersight data
        :param moid: intersight object moid
        :param name: intersight object name
        :return: json http response object
        """
    target_host = urlparse(self.host).netloc
    target_path = urlparse(self.host).path
    query_path = ''
    method = http_method.upper()
    bodyString = ''
    if method not in ['GET', 'POST', 'PATCH', 'DELETE']:
        raise ValueError('Please select a valid HTTP verb (GET/POST/PATCH/DELETE)')
    if resource_path != '' and (not (resource_path, str)):
        raise TypeError('The *resource_path* value is required and must be of type "<str>"')
    if query_params is not None and (not isinstance(query_params, dict)):
        raise TypeError('The *query_params* value must be of type "<dict>"')
    if moid is not None and len(moid.encode('utf-8')) != 24:
        raise ValueError('Invalid *moid* value!')
    if query_params:
        query_path = '?' + urlencode(query_params)
    if method in ('PATCH', 'DELETE'):
        if moid is None:
            if name is not None:
                if isinstance(name, str):
                    moid = self.get_moid_by_name(resource_path, name)
                else:
                    raise TypeError('The *name* value must be of type "<str>"')
            else:
                raise ValueError('Must set either *moid* or *name* with "PATCH/DELETE!"')
    if moid is not None:
        resource_path += '/' + moid
    if method != 'GET':
        bodyString = json.dumps(body)
    target_url = self.host + resource_path + query_path
    request_target = method.lower() + ' ' + target_path + resource_path + query_path
    cdate = get_gmt_date()
    body_digest = get_sha256_digest(bodyString)
    b64_body_digest = b64encode(body_digest.digest())
    auth_header = {'Host': target_host, 'Date': cdate, 'Digest': 'SHA-256=' + b64_body_digest.decode('ascii')}
    string_to_sign = prepare_str_to_sign(request_target, auth_header)
    b64_signed_msg = self.get_sig_b64encode(string_to_sign)
    auth_header = self.get_auth_header(auth_header, b64_signed_msg)
    request_header = {'Accept': 'application/json', 'Content-Type': 'application/json', 'Host': '{0}'.format(target_host), 'Date': '{0}'.format(cdate), 'Digest': 'SHA-256={0}'.format(b64_body_digest.decode('ascii')), 'Authorization': '{0}'.format(auth_header)}
    response, info = fetch_url(self.module, target_url, data=bodyString, headers=request_header, method=method, use_proxy=self.module.params['use_proxy'])
    return (response, info)