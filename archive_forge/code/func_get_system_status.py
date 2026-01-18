from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
import datetime
from ansible.module_utils.six import raise_from
def get_system_status(self):
    params = [{'url': '/cli/global/system/status'}]
    response = self.conn.send_request('get', params)
    if response[0] == 0:
        if 'data' not in response[1]:
            raise AssertionError()
        return response[1]['data']
    return None