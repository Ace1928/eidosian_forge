from __future__ import absolute_import, division, print_function
import time
import json
from ansible.plugins.httpapi import HttpApiBase
from ansible.module_utils.basic import to_text
from ansible_collections.fortinet.fortimanager.plugins.module_utils.common import BASE_HEADERS
from ansible_collections.fortinet.fortimanager.plugins.module_utils.common import FMGBaseException
from ansible_collections.fortinet.fortimanager.plugins.module_utils.common import FMGRCommon
from datetime import datetime
def forticloud_login(self):
    login_data = '{"access_token": "%s"}' % self.get_forticloud_access_token()
    try:
        response, response_data = self.connection.send(path=to_text('/p/forticloud_jsonrpc_login/'), data=to_text(login_data), headers=BASE_HEADERS)
        result = json.loads(to_text(response_data.getvalue()))
        self.log('forticloud login response: %s' % str(self._jsonize(result)))
        return self._set_sid(result)
    except Exception as e:
        raise FMGBaseException(e)