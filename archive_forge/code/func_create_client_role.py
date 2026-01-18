from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def create_client_role(self, rolerep, clientid, realm='master'):
    """ Create a Keycloak client role.

        :param rolerep: a RoleRepresentation of the role to be created. Must contain at minimum the field name.
        :param clientid: Client id for the client role
        :param realm: Realm in which the role resides
        :return: HTTPResponse object on success
        """
    cid = self.get_client_id(clientid, realm=realm)
    if cid is None:
        self.module.fail_json(msg='Could not find client %s in realm %s' % (clientid, realm))
    roles_url = URL_CLIENT_ROLES.format(url=self.baseurl, realm=realm, id=cid)
    try:
        if 'composites' in rolerep:
            keycloak_compatible_composites = self.convert_role_composites(rolerep['composites'])
            rolerep['composites'] = keycloak_compatible_composites
        return open_url(roles_url, method='POST', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, data=json.dumps(rolerep), validate_certs=self.validate_certs)
    except Exception as e:
        self.fail_open_url(e, msg='Could not create role %s for client %s in realm %s: %s' % (rolerep['name'], clientid, realm, str(e)))