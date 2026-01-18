from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def delete_realm_role(self, name, realm='master'):
    """ Delete a realm role.

        :param name: The name of the role.
        :param realm: The realm in which this role resides, default "master".
        """
    role_url = URL_REALM_ROLE.format(url=self.baseurl, realm=realm, name=quote(name, safe=''))
    try:
        return open_url(role_url, method='DELETE', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, validate_certs=self.validate_certs)
    except Exception as e:
        self.fail_open_url(e, msg='Unable to delete role %s in realm %s: %s' % (name, realm, str(e)))