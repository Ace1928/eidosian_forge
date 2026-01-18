from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def create_empty_auth_flow(self, config, realm='master'):
    """
        Create a new empty authentication flow.
        :param config: Representation of the authentication flow to create.
        :param realm: Realm.
        :return: Representation of the new authentication flow.
        """
    try:
        new_flow = dict(alias=config['alias'], providerId=config['providerId'], description=config['description'], topLevel=True)
        open_url(URL_AUTHENTICATION_FLOWS.format(url=self.baseurl, realm=realm), method='POST', http_agent=self.http_agent, headers=self.restheaders, data=json.dumps(new_flow), timeout=self.connection_timeout, validate_certs=self.validate_certs)
        flow_list = json.load(open_url(URL_AUTHENTICATION_FLOWS.format(url=self.baseurl, realm=realm), method='GET', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, validate_certs=self.validate_certs))
        for flow in flow_list:
            if flow['alias'] == config['alias']:
                return flow
        return None
    except Exception as e:
        self.fail_open_url(e, msg='Could not create empty authentication flow %s in realm %s: %s' % (config['alias'], realm, str(e)))