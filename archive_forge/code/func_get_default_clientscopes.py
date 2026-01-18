from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def get_default_clientscopes(self, realm, client_id=None):
    """Fetch the name and ID of all clientscopes on the Keycloak server.

        To fetch the full data of the client scope, make a subsequent call to
        get_clientscope_by_clientscopeid, passing in the ID of the client scope you wish to return.

        :param realm: Realm in which the clientscope resides.
        :param client_id: The client in which the clientscope resides.
        :return The default clientscopes of this realm or client
        """
    url = URL_DEFAULT_CLIENTSCOPES if client_id is None else URL_CLIENT_DEFAULT_CLIENTSCOPES
    return self._get_clientscopes_of_type(realm, url, 'default', client_id)