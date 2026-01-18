from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def get_authentication_flow_by_alias(self, alias, realm='master'):
    """
        Get an authentication flow by it's alias
        :param alias: Alias of the authentication flow to get.
        :param realm: Realm.
        :return: Authentication flow representation.
        """
    try:
        authentication_flow = {}
        authentications = json.load(open_url(URL_AUTHENTICATION_FLOWS.format(url=self.baseurl, realm=realm), method='GET', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, validate_certs=self.validate_certs))
        for authentication in authentications:
            if authentication['alias'] == alias:
                authentication_flow = authentication
                break
        return authentication_flow
    except Exception as e:
        self.fail_open_url(e, msg='Unable get authentication flow %s: %s' % (alias, str(e)))