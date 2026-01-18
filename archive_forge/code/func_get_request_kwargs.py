import copy
import hmac
import base64
import hashlib
from libcloud.utils.py3 import b, httplib, urlquote, urlencode
from libcloud.common.base import JsonResponse, PollingConnection, ConnectionUserAndKey
from libcloud.common.types import ProviderError, MalformedResponseError
from libcloud.compute.types import InvalidCredsError
def get_request_kwargs(self, action, params=None, data='', headers=None, method='GET', context=None):
    command = context['command']
    request_kwargs = {'command': command, 'action': action, 'params': params, 'data': data, 'headers': headers, 'method': method}
    return request_kwargs