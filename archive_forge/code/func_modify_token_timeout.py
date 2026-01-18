from __future__ import absolute_import, division, print_function
import time
from .common import (
from .constants import (
from .icontrol import iControlRestSession
def modify_token_timeout(self, client_session, token_value, token_timeout):
    url = 'https://{0}:{1}/mgmt/shared/authz/tokens/{2}'.format(self.provider['server'], self.provider['server_port'], token_value)
    payload = {'timeout': token_timeout}
    client_session.request.timeout = token_timeout
    response = client_session.patch(url, json=payload)
    if response.status not in [200]:
        raise F5ModuleError(response.content)
    return None