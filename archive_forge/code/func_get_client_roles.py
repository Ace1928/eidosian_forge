from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def get_client_roles(self, clientid, realm='master'):
    """ Obtains role representations for client roles in a specific client

        :param clientid: Client id to be queried
        :param realm: Realm to be queried
        :return: List of dicts of role representations
        """
    cid = self.get_client_id(clientid, realm=realm)
    if cid is None:
        self.module.fail_json(msg='Could not find client %s in realm %s' % (clientid, realm))
    rolelist_url = URL_CLIENT_ROLES.format(url=self.baseurl, realm=realm, id=cid)
    try:
        return json.loads(to_native(open_url(rolelist_url, method='GET', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, validate_certs=self.validate_certs).read()))
    except ValueError as e:
        self.module.fail_json(msg='API returned incorrect JSON when trying to obtain list of roles for client %s in realm %s: %s' % (clientid, realm, str(e)))
    except Exception as e:
        self.fail_open_url(e, msg='Could not obtain list of roles for client %s in realm %s: %s' % (clientid, realm, str(e)))