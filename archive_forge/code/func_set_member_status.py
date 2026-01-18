from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six import iteritems
def set_member_status(self, values):
    """ Sets a balancer member's status attributes amongst pre-mapped values."""
    values_mapping = {'disabled': '&w_status_D', 'drained': '&w_status_N', 'hot_standby': '&w_status_H', 'ignore_errors': '&w_status_I'}
    request_body = regexp_extraction(self.management_url, EXPRESSION, 1)
    values_url = ''.join(('{0}={1}'.format(url_param, 1 if values[mode] else 0) for mode, url_param in iteritems(values_mapping)))
    request_body = '{0}{1}'.format(request_body, values_url)
    response = fetch_url(self.module, self.management_url, data=request_body)
    if response[1]['status'] != 200:
        self.module.fail_json(msg='Could not set the member status! ' + self.host + ' ' + response[1]['status'])