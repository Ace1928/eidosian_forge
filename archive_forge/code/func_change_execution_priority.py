from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def change_execution_priority(self, executionId, diff, realm='master'):
    """ Raise or lower execution priority of diff time

        :param executionId: id of execution to lower priority
        :param realm: realm the client is in
        :param diff: Integer number, raise of diff time if positive lower of diff time if negative
        :return: HTTPResponse object on success
        """
    try:
        if diff > 0:
            for i in range(diff):
                open_url(URL_AUTHENTICATION_EXECUTION_RAISE_PRIORITY.format(url=self.baseurl, realm=realm, id=executionId), method='POST', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, validate_certs=self.validate_certs)
        elif diff < 0:
            for i in range(-diff):
                open_url(URL_AUTHENTICATION_EXECUTION_LOWER_PRIORITY.format(url=self.baseurl, realm=realm, id=executionId), method='POST', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, validate_certs=self.validate_certs)
    except Exception as e:
        self.fail_open_url(e, msg='Unable to change execution priority %s: %s' % (executionId, str(e)))