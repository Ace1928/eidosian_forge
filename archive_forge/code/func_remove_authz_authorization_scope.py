from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def remove_authz_authorization_scope(self, id, client_id, realm):
    """Remove an authorization scope from a Keycloak client"""
    url = URL_AUTHZ_AUTHORIZATION_SCOPE.format(url=self.baseurl, id=id, client_id=client_id, realm=realm)
    try:
        return open_url(url, method='DELETE', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, validate_certs=self.validate_certs)
    except Exception as e:
        self.fail_open_url(e, msg='Could not delete scope %s for client %s in realm %s: %s' % (id, client_id, realm, str(e)))