from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def get_client_user_available_rolemappings(self, uid, cid, realm='master'):
    """ Fetch the available role of a client for a specified user on the Keycloak server.

        :param uid: ID of the user from which to obtain the rolemappings.
        :param cid: ID of the client from which to obtain the rolemappings.
        :param realm: Realm from which to obtain the rolemappings.
        :return: The effective rollemappings of specified client and user of the realm (default "master").
        """
    available_rolemappings_url = URL_CLIENT_USER_ROLEMAPPINGS_AVAILABLE.format(url=self.baseurl, realm=realm, id=uid, client=cid)
    try:
        return json.loads(to_native(open_url(available_rolemappings_url, method='GET', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, validate_certs=self.validate_certs).read()))
    except Exception as e:
        self.fail_open_url(e, msg='Could not fetch effective rolemappings for client %s and user %s, realm %s: %s' % (cid, uid, realm, str(e)))