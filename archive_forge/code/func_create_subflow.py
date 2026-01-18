from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def create_subflow(self, subflowName, flowAlias, realm='master', flowType='basic-flow'):
    """ Create new sublow on the flow

        :param subflowName: name of the subflow to create
        :param flowAlias: name of the parent flow
        :return: HTTPResponse object on success
        """
    try:
        newSubFlow = {}
        newSubFlow['alias'] = subflowName
        newSubFlow['provider'] = 'registration-page-form'
        newSubFlow['type'] = flowType
        open_url(URL_AUTHENTICATION_FLOW_EXECUTIONS_FLOW.format(url=self.baseurl, realm=realm, flowalias=quote(flowAlias, safe='')), method='POST', http_agent=self.http_agent, headers=self.restheaders, data=json.dumps(newSubFlow), timeout=self.connection_timeout, validate_certs=self.validate_certs)
    except Exception as e:
        self.fail_open_url(e, msg='Unable to create new subflow %s: %s' % (subflowName, str(e)))