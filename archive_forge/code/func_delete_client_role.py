from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def delete_client_role(self, name, clientid, realm='master'):
    """ Delete a role. One of name or roleid must be provided.

        :param name: The name of the role.
        :param clientid: Client id for the client role
        :param realm: Realm in which the role resides
        """
    cid = self.get_client_id(clientid, realm=realm)
    if cid is None:
        self.module.fail_json(msg='Could not find client %s in realm %s' % (clientid, realm))
    role_url = URL_CLIENT_ROLE.format(url=self.baseurl, realm=realm, id=cid, name=quote(name, safe=''))
    try:
        return open_url(role_url, method='DELETE', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, validate_certs=self.validate_certs)
    except Exception as e:
        self.fail_open_url(e, msg='Unable to delete role %s for client %s in realm %s: %s' % (name, clientid, realm, str(e)))