from __future__ import (absolute_import, division, print_function)
import time
import json
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.errors import AnsibleError
from ansible.plugins.lookup import LookupBase
from ansible.utils.display import Display
def encode_jwt(app_id, jwk, exp=600):
    now = int(time.time())
    payload = {'iat': now, 'exp': now + exp, 'iss': app_id}
    try:
        return jwt_instance.encode(payload, jwk, alg='RS256')
    except Exception as e:
        raise AnsibleError('Error while encoding jwt: {0}'.format(e))