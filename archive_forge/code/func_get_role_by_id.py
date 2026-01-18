from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def get_role_by_id(self, rid, realm='master'):
    """ Fetch a role by its id on the Keycloak server.

        :param rid: ID of the role.
        :param realm: Realm from which to obtain the rolemappings.
        :return: The role.
        """
    client_roles_url = URL_ROLES_BY_ID.format(url=self.baseurl, realm=realm, id=rid)
    try:
        return json.loads(to_native(open_url(client_roles_url, method='GET', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, validate_certs=self.validate_certs).read()))
    except Exception as e:
        self.fail_open_url(e, msg='Could not fetch role for id %s in realm %s: %s' % (rid, realm, str(e)))