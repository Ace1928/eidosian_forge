import time
import hashlib
from typing import List
from libcloud.http import LibcloudConnection
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import InvalidCredsError
from libcloud.utils.connection import get_response_object
def request_consumer_key(self, user_id):
    action = self.request_path + '/auth/credential'
    data = json.dumps({'accessRules': DEFAULT_ACCESS_RULES, 'redirection': 'http://ovh.com'})
    headers = {'Content-Type': 'application/json', 'X-Ovh-Application': user_id}
    httpcon = LibcloudConnection(host=self.host, port=443)
    try:
        httpcon.request(method='POST', url=action, body=data, headers=headers)
    except Exception as e:
        handle_and_rethrow_user_friendly_invalid_region_error(host=self.host, e=e)
    response = OvhResponse(httpcon.getresponse(), httpcon)
    if response.status == httplib.UNAUTHORIZED:
        raise InvalidCredsError()
    json_response = response.parse_body()
    httpcon.close()
    return json_response