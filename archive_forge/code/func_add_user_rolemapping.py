from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def add_user_rolemapping(self, uid, cid, role_rep, realm='master'):
    """ Assign a realm or client role to a specified user on the Keycloak server.

        :param uid: ID of the user roles are assigned to.
        :param cid: ID of the client from which to obtain the rolemappings. If empty, roles are from the realm
        :param role_rep: Representation of the role to assign.
        :param realm: Realm from which to obtain the rolemappings.
        :return: None.
        """
    if cid is None:
        user_realm_rolemappings_url = URL_REALM_ROLEMAPPINGS.format(url=self.baseurl, realm=realm, id=uid)
        try:
            open_url(user_realm_rolemappings_url, method='POST', http_agent=self.http_agent, headers=self.restheaders, data=json.dumps(role_rep), validate_certs=self.validate_certs, timeout=self.connection_timeout)
        except Exception as e:
            self.fail_open_url(e, msg='Could not map roles to userId %s for realm %s and roles %s: %s' % (uid, realm, json.dumps(role_rep), str(e)))
    else:
        user_client_rolemappings_url = URL_CLIENT_USER_ROLEMAPPINGS.format(url=self.baseurl, realm=realm, id=uid, client=cid)
        try:
            open_url(user_client_rolemappings_url, method='POST', http_agent=self.http_agent, headers=self.restheaders, data=json.dumps(role_rep), validate_certs=self.validate_certs, timeout=self.connection_timeout)
        except Exception as e:
            self.fail_open_url(e, msg='Could not map roles to userId %s for client %s, realm %s and roles %s: %s' % (cid, uid, realm, json.dumps(role_rep), str(e)))