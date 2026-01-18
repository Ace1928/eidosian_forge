import time
from libcloud.http import LibcloudConnection
from libcloud.utils.py3 import urlparse, urlencode, basestring
from libcloud.common.base import BaseDriver, RawResponse, JsonResponse, ConnectionUserAndKey
def get_token_from_credentials(self):
    """
        Log in and get bearer token used to authorize API requests.
        """
    conn = self.conn_class(self.login_host, 443, timeout=self.timeout)
    conn.connect()
    params = urlencode({'grant_type': 'client_credentials', 'client_id': self.user_id, 'client_secret': self.key, 'resource': self.login_resource})
    headers = {'Content-type': 'application/x-www-form-urlencoded'}
    conn.request('POST', '/%s/oauth2/token' % self.tenant_id, params, headers)
    js = AzureAuthJsonResponse(conn.getresponse(), conn)
    self.access_token = js.object['access_token']
    self.expires_on = js.object['expires_on']