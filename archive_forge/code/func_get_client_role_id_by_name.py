from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def get_client_role_id_by_name(self, cid, name, realm='master'):
    """ Get the role ID of a client.

        :param cid: ID of the client from which to obtain the rolemappings.
        :param name: Name of the role.
        :param realm: Realm from which to obtain the rolemappings.
        :return: The ID of the role, None if not found.
        """
    rolemappings = self.get_client_roles_by_id(cid, realm=realm)
    for role in rolemappings:
        if name == role['name']:
            return role['id']
    return None