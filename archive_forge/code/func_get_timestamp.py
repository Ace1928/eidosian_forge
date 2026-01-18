import time
import hashlib
from typing import List
from libcloud.http import LibcloudConnection
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import InvalidCredsError
from libcloud.utils.connection import get_response_object
def get_timestamp(self):
    if not self._timedelta:
        url = 'https://{}{}/auth/time'.format(self.host, API_ROOT)
        response = get_response_object(url=url, method='GET', headers={})
        if not response or not response.body:
            raise Exception('Failed to get current time from Ovh API')
        timestamp = int(response.body)
        self._timedelta = timestamp - int(time.time())
    return int(time.time()) + self._timedelta