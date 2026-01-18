from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def create_clientscope(self, clientscoperep, realm='master'):
    """ Create a Keycloak clientscope.

        :param clientscoperep: a ClientScopeRepresentation of the clientscope to be created. Must contain at minimum the field name.
        :return: HTTPResponse object on success
        """
    clientscopes_url = URL_CLIENTSCOPES.format(url=self.baseurl, realm=realm)
    try:
        return open_url(clientscopes_url, method='POST', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, data=json.dumps(clientscoperep), validate_certs=self.validate_certs)
    except Exception as e:
        self.fail_open_url(e, msg='Could not create clientscope %s in realm %s: %s' % (clientscoperep['name'], realm, str(e)))