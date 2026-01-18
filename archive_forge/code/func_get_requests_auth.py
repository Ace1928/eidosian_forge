import abc
import requests
import requests.auth
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1.identity import v3
def get_requests_auth(self):
    return requests.auth.HTTPBasicAuth(self.username, self.password)