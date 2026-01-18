from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def get_client_roles_by_id(self, cid, realm='master'):
    """ Fetch the roles of the a client on the Keycloak server.

        :param cid: ID of the client from which to obtain the rolemappings.
        :param realm: Realm from which to obtain the rolemappings.
        :return: The rollemappings of specified group and client of the realm (default "master").
        """
    client_roles_url = URL_CLIENT_ROLES.format(url=self.baseurl, realm=realm, id=cid)
    try:
        return json.loads(to_native(open_url(client_roles_url, method='GET', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, validate_certs=self.validate_certs).read()))
    except Exception as e:
        self.fail_open_url(e, msg='Could not fetch rolemappings for client %s in realm %s: %s' % (cid, realm, str(e)))