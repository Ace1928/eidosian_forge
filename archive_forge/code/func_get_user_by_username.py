from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def get_user_by_username(self, username, realm='master'):
    """ Fetch a keycloak user within a realm based on its username.

        If the user does not exist, None is returned.
        :param username: Username of the user to fetch.
        :param realm: Realm in which the user resides; default 'master'
        """
    users_url = URL_USERS.format(url=self.baseurl, realm=realm)
    users_url += '?username=%s&exact=true' % username
    try:
        userrep = None
        users = json.loads(to_native(open_url(users_url, method='GET', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, validate_certs=self.validate_certs).read()))
        for user in users:
            if user['username'] == username:
                userrep = user
                break
        return userrep
    except ValueError as e:
        self.module.fail_json(msg='API returned incorrect JSON when trying to obtain the user for realm %s and username %s: %s' % (realm, username, str(e)))
    except Exception as e:
        self.fail_open_url(e, msg='Could not obtain the user for realm %s and username %s: %s' % (realm, username, str(e)))