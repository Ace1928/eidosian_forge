import time
import hashlib
from typing import List
from libcloud.http import LibcloudConnection
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import InvalidCredsError
from libcloud.utils.connection import get_response_object
def make_signature(self, method, action, params, data, timestamp):
    full_url = 'https://{}{}'.format(self.host, action)
    if params:
        full_url += '?'
        for key, value in params.items():
            full_url += '{}={}&'.format(key, value)
        full_url = full_url[:-1]
    sha1 = hashlib.sha1()
    base_signature = '+'.join([self.key, self.consumer_key, method.upper(), full_url, data if data else '', str(timestamp)])
    sha1.update(base_signature.encode())
    signature = '$1$' + sha1.hexdigest()
    return signature