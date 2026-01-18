from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def register_required_action(self, rep, realm='master'):
    """
        Register required action.
        :param rep:   JSON containing 'providerId', and 'name' attributes.
        :param realm: Realm name (not id).
        :return:      Representation of the required action.
        """
    data = {'name': rep['name'], 'providerId': rep['providerId']}
    try:
        return open_url(URL_AUTHENTICATION_REGISTER_REQUIRED_ACTION.format(url=self.baseurl, realm=realm), method='POST', http_agent=self.http_agent, headers=self.restheaders, data=json.dumps(data), timeout=self.connection_timeout, validate_certs=self.validate_certs)
    except Exception as e:
        self.fail_open_url(e, msg='Unable to register required action %s in realm %s: %s' % (rep['name'], realm, str(e)))